#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#include "gpulz.cuh"

__global__ void pixel2float(const uint8_t *input, float *output, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        output[idx] = input[idx] / 255.0f;
    }
}

__global__ void quantize(const float *input, int8_t *output, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        output[idx] = int8_t(input[idx]);
    }
}

__global__ void dequantize(const int8_t *input, float *output, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        output[idx] = float(input[idx]);
    }
}

void compress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath, uint32_t x, uint32_t y)
{
    std::cout << "Compressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, gpuIdx);
    at::cuda::setCurrentCUDAStream(myStream);

    cudaStream_t stream = myStream.stream();

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    auto modelLoadStart = std::chrono::high_resolution_clock::now();
    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath, deviceString);
        // module.to(torch::kHalf);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }
    cudaStreamSynchronize(stream);
    auto modelLoadEnd = std::chrono::high_resolution_clock::now();

    size_t binaryFileSize = io::fileSize(inputPath);
    uint8_t *hInputArr;
    cudaMallocHost(&hInputArr, binaryFileSize);

    uint8_t *dInputArr;
    cudaMallocAsync(&dInputArr, binaryFileSize, stream);

    auto readStart = std::chrono::high_resolution_clock::now();
    io::read_binary_to_array<uint8_t>(inputPath, hInputArr, binaryFileSize);

    cudaMemcpyAsync(dInputArr, hInputArr, binaryFileSize, cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);
    auto readEnd = std::chrono::high_resolution_clock::now();

    float *dInputArrFloat;
    cudaMallocAsync(&dInputArrFloat, binaryFileSize * sizeof(float), stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (binaryFileSize + threadsPerBlock - 1) / threadsPerBlock;

    pixel2float<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInputArr, dInputArrFloat, binaryFileSize);

    cudaStreamSynchronize(stream);

    cudaFreeAsync(dInputArr, stream);

    // uint32_t x = 2000;
    // uint32_t y = 2000;

    auto dims = torch::IntArrayRef{1, 3, x, y};
    auto inputTensor = torch::from_blob(dInputArrFloat, dims, deviceString);
    // inputTensor.to(torch::kHalf);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inferenceInputVec;
    inferenceInputVec.push_back(inputTensor);

    auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the model and turn its output into a tensor.
    at::Tensor outputTensor = module.forward(inferenceInputVec).toTensor();
    cudaStreamSynchronize(stream);
    auto inferenceEnd = std::chrono::high_resolution_clock::now();

    auto dInferenceOutput = outputTensor.data_ptr<float>();
    auto dInferenceOutputSize = outputTensor.sizes();
    size_t dInferenceOutputSizeTotal = dInferenceOutputSize[0] * dInferenceOutputSize[1] * dInferenceOutputSize[2] * dInferenceOutputSize[3];

    int8_t *dQuantizedOutput;
    cudaMallocAsync(&dQuantizedOutput, dInferenceOutputSizeTotal * sizeof(uint8_t), stream);

    // Kernel launch parameters
    threadsPerBlock = 256;
    blocksPerGrid = (dInferenceOutputSizeTotal + threadsPerBlock - 1) / threadsPerBlock;

    quantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInferenceOutput, dQuantizedOutput, dInferenceOutputSizeTotal);

    uint32_t *dToWriteBack;
    cudaMallocAsync(&dToWriteBack, sizeof(uint8_t) * (dInferenceOutputSizeTotal / 4 + 1) * 4, stream);

    cudaMemcpyAsync(dToWriteBack, &x, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 1, &y, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(dToWriteBack + 2, &dInferenceOutputSize[1], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 3, &dInferenceOutputSize[2], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 4, &dInferenceOutputSize[3], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    uint32_t gpulzCompedSize = 0;

    auto gpulzStart = std::chrono::high_resolution_clock::now();
    gpulz::sparseGpulzComp(dQuantizedOutput, dToWriteBack, &gpulzCompedSize, dInferenceOutputSizeTotal, gpuIdx, stream);
    cudaStreamSynchronize(stream);
    auto gpulzEnd = std::chrono::high_resolution_clock::now();

    uint8_t *hOutput;
    cudaMallocHost(&hOutput, gpulzCompedSize + 5 * sizeof(uint32_t));
    cudaMemcpyAsync(hOutput, dToWriteBack, gpulzCompedSize + 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    auto writeStart = std::chrono::high_resolution_clock::now();
    io::write_array_to_binary<uint8_t>(outputPath, hOutput, gpulzCompedSize + 5 * sizeof(uint32_t));
    auto writeEnd = std::chrono::high_resolution_clock::now();

    cudaFreeHost(hInputArr);
    cudaFreeHost(hOutput);

    cudaFree(dInputArrFloat);
    cudaFree(dQuantizedOutput);
    cudaFree(dToWriteBack);

    auto end = std::chrono::high_resolution_clock::now();

    float compressionRatio = (float)binaryFileSize / ((float)gpulzCompedSize + 5.0f * sizeof(uint32_t));
    std::cout << "Compression ratio: " << compressionRatio << std::endl;

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto readDuration = std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart);

    // Output the time elapsed
    std::cout << "Time taken for readDuration: " << readDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto modelLoadDuration = std::chrono::duration_cast<std::chrono::microseconds>(modelLoadEnd - modelLoadStart);

    // Output the time elapsed
    std::cout << "Time taken for modelLoadDuration: " << modelLoadDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto inferenceDuration = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart);

    // Output the time elapsed
    std::cout << "Time taken for inferenceDuration: " << inferenceDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto gpulzDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpulzEnd - gpulzStart);

    // Output the time elapsed
    std::cout << "Time taken for gpulzDuration: " << gpulzDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto writeDuration = std::chrono::duration_cast<std::chrono::microseconds>(writeEnd - writeStart);

    // Output the time elapsed
    std::cout << "Time taken for writeDuration: " << writeDuration.count() << " microseconds." << std::endl;

    return;
}

void decompress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath)
{
    std::cout << "Decompressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, gpuIdx);
    at::cuda::setCurrentCUDAStream(myStream);

    cudaStream_t stream = myStream.stream();

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    uint32_t x = 0;
    uint32_t y = 0;

    uint32_t dInferenceOutputSize0 = 0;
    uint32_t dInferenceOutputSize1 = 0;
    uint32_t dInferenceOutputSize2 = 0;

    uint32_t *hInputArr;
    size_t binaryFileSize = io::fileSize(inputPath);
    cudaMallocHost(&hInputArr, binaryFileSize);
    io::read_binary_to_array<uint32_t>(inputPath, hInputArr, binaryFileSize / sizeof(uint32_t));

    x = hInputArr[0];
    y = hInputArr[1];

    dInferenceOutputSize0 = hInputArr[2];
    dInferenceOutputSize1 = hInputArr[3];
    dInferenceOutputSize2 = hInputArr[4];

    uint8_t *dCompressedOutput;
    cudaMallocAsync(&dCompressedOutput, binaryFileSize - sizeof(uint32_t) * 5, stream);
    cudaMemcpyAsync(dCompressedOutput, hInputArr + 5, binaryFileSize - sizeof(uint32_t) * 5, cudaMemcpyHostToDevice, stream);

    INPUT_TYPE *dDecompedOutput;
    cudaMallocAsync(&dDecompedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(INPUT_TYPE), stream);

    gpulz::decompress(dCompressedOutput, dDecompedOutput, gpuIdx, stream);

    float *dDequantizedOutput;
    cudaMallocAsync(&dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(float), stream);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 + threadsPerBlock - 1) / threadsPerBlock;

    dequantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dDecompedOutput, dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2);

    cudaStreamSynchronize(stream);

    auto dims = torch::IntArrayRef{1, dInferenceOutputSize0, dInferenceOutputSize1, dInferenceOutputSize2};
    auto inputTensor = torch::from_blob(dDequantizedOutput, dims, deviceString);

    torch::jit::script::Module synthesisModel;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        synthesisModel = torch::jit::load(modelPath, deviceString);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> synthesisModelInputs;
    synthesisModelInputs.push_back(inputTensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor tmp_output = synthesisModel.forward(synthesisModelInputs).toTensor();

    auto dSynthesisOutput = tmp_output.data_ptr<float>();

    float *hOutput;
    cudaMallocHost(&hOutput, 3 * x * y * sizeof(float));
    cudaMemcpyAsync(hOutput, dSynthesisOutput, 3 * x * y * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    io::write_array_to_binary<float>(outputPath, hOutput, 3 * x * y);

    cudaFreeHost(hInputArr);
    cudaFreeHost(hOutput);

    cudaFree(dCompressedOutput);
    cudaFree(dDecompedOutput);
    cudaFree(dDequantizedOutput);

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    return;
}

void gdsCompress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath, uint32_t x, uint32_t y)
{
    std::cout << "Compressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    // at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, gpuIdx);
    // cudaStream_t stream = myStream.stream();

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    at::cuda::CUDAStream myStream = at::cuda::getStreamFromExternal(stream, gpuIdx);
    at::cuda::setCurrentCUDAStream(myStream);

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    auto cufileDriverOpenStart = std::chrono::high_resolution_clock::now();
    CUfileError_t status;
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS)
    {
        std::cerr << "cufile driver open error" << std::endl;
        return;
    }
    auto cufileDriverOpenEnd = std::chrono::high_resolution_clock::now();

    auto modelLoadStart = std::chrono::high_resolution_clock::now();
    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath, deviceString);
        // module.to(torch::kHalf);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }
    cudaStreamSynchronize(stream);
    auto modelLoadEnd = std::chrono::high_resolution_clock::now();

    size_t binaryFileSize = 3 * x * y * sizeof(INPUT_TYPE);
    // uint8_t *hInputArr;
    // cudaMallocHost(&hInputArr, binaryFileSize);

    // auto readStart = std::chrono::high_resolution_clock::now();
    // io::read_binary_to_array<uint8_t>(inputPath, hInputArr, binaryFileSize);
    // auto readEnd = std::chrono::high_resolution_clock::now();

    uint8_t *dInputArr;
    cudaMallocAsync(&dInputArr, binaryFileSize, stream);

    auto readStart = std::chrono::high_resolution_clock::now();
    io::cufileReadAsync(inputPath.c_str(), dInputArr, &binaryFileSize, stream);
    auto readEnd = std::chrono::high_resolution_clock::now();

    float *dInputArrFloat;
    cudaMallocAsync(&dInputArrFloat, binaryFileSize * sizeof(float), stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (binaryFileSize + threadsPerBlock - 1) / threadsPerBlock;

    pixel2float<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInputArr, dInputArrFloat, binaryFileSize);

    cudaStreamSynchronize(stream);

    cudaFreeAsync(dInputArr, stream);

    // uint32_t x = 2000;
    // uint32_t y = 2000;

    auto dims = torch::IntArrayRef{1, 3, x, y};
    auto inputTensor = torch::from_blob(dInputArrFloat, dims, deviceString);
    // inputTensor.to(torch::kHalf);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inferenceInputVec;
    inferenceInputVec.push_back(inputTensor);

    auto inferenceStart = std::chrono::high_resolution_clock::now();
    // Execute the model and turn its output into a tensor.
    at::Tensor outputTensor = module.forward(inferenceInputVec).toTensor();
    cudaStreamSynchronize(stream);
    auto inferenceEnd = std::chrono::high_resolution_clock::now();

    auto dInferenceOutput = outputTensor.data_ptr<float>();
    auto dInferenceOutputSize = outputTensor.sizes();
    size_t dInferenceOutputSizeTotal = dInferenceOutputSize[0] * dInferenceOutputSize[1] * dInferenceOutputSize[2] * dInferenceOutputSize[3];

    int8_t *dQuantizedOutput;
    cudaMallocAsync(&dQuantizedOutput, dInferenceOutputSizeTotal * sizeof(uint8_t), stream);

    // Kernel launch parameters
    threadsPerBlock = 256;
    blocksPerGrid = (dInferenceOutputSizeTotal + threadsPerBlock - 1) / threadsPerBlock;

    quantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInferenceOutput, dQuantizedOutput, dInferenceOutputSizeTotal);

    uint32_t *dToWriteBack;
    cudaMallocAsync(&dToWriteBack, sizeof(uint8_t) * (dInferenceOutputSizeTotal / 4 + 1) * 4, stream);

    cudaMemcpyAsync(dToWriteBack, &x, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 1, &y, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(dToWriteBack + 2, &dInferenceOutputSize[1], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 3, &dInferenceOutputSize[2], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dToWriteBack + 4, &dInferenceOutputSize[3], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    uint32_t gpulzCompedSize = 0;

    auto gpulzStart = std::chrono::high_resolution_clock::now();
    gpulz::sparseGpulzComp(dQuantizedOutput, dToWriteBack, &gpulzCompedSize, dInferenceOutputSizeTotal, gpuIdx, stream);
    cudaStreamSynchronize(stream);
    auto gpulzEnd = std::chrono::high_resolution_clock::now();

    // uint8_t *hOutput;
    // cudaMallocHost(&hOutput, gpulzCompedSize + 5 * sizeof(uint32_t));
    // cudaMemcpyAsync(hOutput, dToWriteBack, gpulzCompedSize + 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    // cudaStreamSynchronize(stream);

    // auto writeStart = std::chrono::high_resolution_clock::now();
    // io::write_array_to_binary<uint8_t>(outputPath, hOutput, gpulzCompedSize + 5 * sizeof(uint32_t));
    // auto writeEnd = std::chrono::high_resolution_clock::now();
    size_t tmpWriteSize = gpulzCompedSize + 5 * sizeof(uint32_t);

    auto writeStart = std::chrono::high_resolution_clock::now();
    io::cufileWriteAsync(outputPath.c_str(), dToWriteBack, &tmpWriteSize, stream);
    auto writeEnd = std::chrono::high_resolution_clock::now();

    cudaFree(dInputArrFloat);
    cudaFree(dQuantizedOutput);
    cudaFree(dToWriteBack);

    auto end = std::chrono::high_resolution_clock::now();

    float compressionRatio = (float)binaryFileSize / ((float)gpulzCompedSize + 5.0f * sizeof(uint32_t));
    std::cout << "Compression ratio: " << compressionRatio << std::endl;

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto readDuration = std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart);

    // Output the time elapsed
    std::cout << "Time taken for readDuration: " << readDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto modelLoadDuration = std::chrono::duration_cast<std::chrono::microseconds>(modelLoadEnd - modelLoadStart);

    // Output the time elapsed
    std::cout << "Time taken for modelLoadDuration: " << modelLoadDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto inferenceDuration = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart);

    // Output the time elapsed
    std::cout << "Time taken for inferenceDuration: " << inferenceDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto gpulzDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpulzEnd - gpulzStart);

    // Output the time elapsed
    std::cout << "Time taken for gpulzDuration: " << gpulzDuration.count() << " microseconds." << std::endl;

    // Calculate elapsed time in microseconds
    auto writeDuration = std::chrono::duration_cast<std::chrono::microseconds>(writeEnd - writeStart);

    // Output the time elapsed
    std::cout << "Time taken for writeDuration: " << writeDuration.count() << " microseconds." << std::endl;

    auto cufileDriverOpenDuration = std::chrono::duration_cast<std::chrono::microseconds>(cufileDriverOpenEnd - cufileDriverOpenStart);
    std::cout << "Time taken for cufileDriverOpenDuration: " << cufileDriverOpenDuration.count() << " microseconds." << std::endl;


    return;
}

void gdsDecompress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath)
{
    std::cout << "Decompressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, gpuIdx);
    at::cuda::setCurrentCUDAStream(myStream);

    cudaStream_t stream = myStream.stream();

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    auto cufileDriverOpenStart = std::chrono::high_resolution_clock::now();
    CUfileError_t status;
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS)
    {
        std::cerr << "cufile driver open error" << std::endl;
        return;
    }
    auto cufileDriverOpenEnd = std::chrono::high_resolution_clock::now();

    uint32_t x = 0;
    uint32_t y = 0;

    uint32_t dInferenceOutputSize0 = 0;
    uint32_t dInferenceOutputSize1 = 0;
    uint32_t dInferenceOutputSize2 = 0;

    size_t binaryFileSize = io::fileSize(inputPath);
    uint32_t *dRawInput;
    cudaMallocAsync(&dRawInput, binaryFileSize, stream);

    uint8_t *dCompressedOutput = (uint8_t *)(dRawInput + 5);

    auto readStart = std::chrono::high_resolution_clock::now();
    io::cufileReadAsync(inputPath.c_str(), dRawInput, &binaryFileSize, stream);
    auto readEnd = std::chrono::high_resolution_clock::now();

    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(&x, dRawInput, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&y, dRawInput + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    cudaMemcpyAsync(&dInferenceOutputSize0, dRawInput + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&dInferenceOutputSize1, dRawInput + 3, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&dInferenceOutputSize2, dRawInput + 4, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // uint32_t *hInputArr;
    // size_t binaryFileSize = io::fileSize(inputPath);
    // cudaMallocHost(&hInputArr, binaryFileSize);
    // io::read_binary_to_array<uint32_t>(inputPath, hInputArr, binaryFileSize / sizeof(uint32_t));

    // x = hInputArr[0];
    // y = hInputArr[1];

    // dInferenceOutputSize0 = hInputArr[2];
    // dInferenceOutputSize1 = hInputArr[3];
    // dInferenceOutputSize2 = hInputArr[4];

    // uint8_t *dCompressedOutput;
    // cudaMallocAsync(&dCompressedOutput, binaryFileSize - sizeof(uint32_t) * 5, stream);
    // cudaMemcpyAsync(dCompressedOutput, hInputArr + 5, binaryFileSize - sizeof(uint32_t) * 5, cudaMemcpyHostToDevice, stream);

    INPUT_TYPE *dDecompedOutput;
    cudaMallocAsync(&dDecompedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(INPUT_TYPE), stream);

    gpulz::decompress(dCompressedOutput, dDecompedOutput, gpuIdx, stream);

    float *dDequantizedOutput;
    cudaMallocAsync(&dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(float), stream);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 + threadsPerBlock - 1) / threadsPerBlock;

    dequantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dDecompedOutput, dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2);

    cudaStreamSynchronize(stream);

    auto dims = torch::IntArrayRef{1, dInferenceOutputSize0, dInferenceOutputSize1, dInferenceOutputSize2};
    auto inputTensor = torch::from_blob(dDequantizedOutput, dims, deviceString);

    torch::jit::script::Module synthesisModel;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        synthesisModel = torch::jit::load(modelPath, deviceString);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> synthesisModelInputs;
    synthesisModelInputs.push_back(inputTensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor tmp_output = synthesisModel.forward(synthesisModelInputs).toTensor();

    auto dSynthesisOutput = tmp_output.data_ptr<float>();

    // float *hOutput;
    // cudaMallocHost(&hOutput, 3 * x * y * sizeof(float));
    // cudaMemcpyAsync(hOutput, dSynthesisOutput, 3 * x * y * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // io::write_array_to_binary<float>(outputPath, hOutput, 3 * x * y);
    size_t tmpWriteSize = 3 * x * y * sizeof(float);

    auto writeStart = std::chrono::high_resolution_clock::now();
    io::cufileWriteAsync(outputPath.c_str(), dSynthesisOutput, &tmpWriteSize, stream);
    auto writeEnd = std::chrono::high_resolution_clock::now();

    // cudaFreeHost(hInputArr);
    // cudaFreeHost(hOutput);

    cudaFree(dCompressedOutput);
    cudaFree(dDecompedOutput);
    cudaFree(dDequantizedOutput);

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    return;
}

void ompCompress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath, uint32_t x, uint32_t y)
{
    std::cout << "Compressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    at::cuda::CUDAStream *myStream;
    cudaMallocHost(&myStream, 4 * sizeof(at::cuda::CUDAStream));

    for (int i = 0; i < 4; i++)
    {
        myStream[i] = at::cuda::getStreamFromPool(false, gpuIdx);
    }

    at::cuda::setCurrentCUDAStream(myStream[0]);

    // cudaStream_t stream = myStream[0].stream();

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath, deviceString);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }

    uint32_t gpulzCompedSize[4];

    std::string inputPathArr[4];
    std::string outputPathArr[4];
    for (int i = 0; i < 4; i++)
    {
        inputPathArr[i] = inputPath + "_" + std::to_string(i) + ".bin";
        outputPathArr[i] = outputPath + "_" + std::to_string(i) + ".comp";
    }

    size_t binaryFileSize = 3 * x * y * sizeof(INPUT_TYPE);

#pragma omp parallel num_threads(4)
    {
        int threadIdx = omp_get_thread_num();
        // printf("Thread %d\n", threadIdx);

        at::cuda::setCurrentCUDAStream(myStream[threadIdx]);

        cudaStream_t stream = myStream[threadIdx].stream();

        uint8_t *hInputArr;
        cudaMallocHost(&hInputArr, binaryFileSize);
        io::read_binary_to_array<uint8_t>(inputPathArr[threadIdx], hInputArr, binaryFileSize);

        uint8_t *dInputArr;
        cudaMallocAsync(&dInputArr, binaryFileSize, stream);
        cudaMemcpyAsync(dInputArr, hInputArr, binaryFileSize, cudaMemcpyHostToDevice, stream);

        float *dInputArrFloat;
        cudaMallocAsync(&dInputArrFloat, binaryFileSize * sizeof(float), stream);

        int threadsPerBlock = 256;
        int blocksPerGrid = (binaryFileSize + threadsPerBlock - 1) / threadsPerBlock;

        pixel2float<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInputArr, dInputArrFloat, binaryFileSize);

        cudaStreamSynchronize(stream);

        cudaFreeAsync(dInputArr, stream);

        auto dims = torch::IntArrayRef{1, 3, x, y};
        auto inputTensor = torch::from_blob(dInputArrFloat, dims, deviceString);

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inferenceInputVec;
        inferenceInputVec.push_back(inputTensor);

        // Execute the model and turn its output into a tensor.
        at::Tensor outputTensor = module.forward(inferenceInputVec).toTensor();
        cudaStreamSynchronize(stream);

        auto dInferenceOutput = outputTensor.data_ptr<float>();
        auto dInferenceOutputSize = outputTensor.sizes();
        size_t dInferenceOutputSizeTotal = dInferenceOutputSize[0] * dInferenceOutputSize[1] * dInferenceOutputSize[2] * dInferenceOutputSize[3];

        int8_t *dQuantizedOutput;
        cudaMallocAsync(&dQuantizedOutput, dInferenceOutputSizeTotal * sizeof(uint8_t), stream);

        // Kernel launch parameters
        threadsPerBlock = 256;
        blocksPerGrid = (dInferenceOutputSizeTotal + threadsPerBlock - 1) / threadsPerBlock;

        quantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dInferenceOutput, dQuantizedOutput, dInferenceOutputSizeTotal);

        uint32_t *dToWriteBack;
        cudaMallocAsync(&dToWriteBack, sizeof(uint8_t) * (dInferenceOutputSizeTotal / 4 + 1) * 4, stream);

        cudaMemcpyAsync(dToWriteBack, &x, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dToWriteBack + 1, &y, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(dToWriteBack + 2, &dInferenceOutputSize[1], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dToWriteBack + 3, &dInferenceOutputSize[2], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dToWriteBack + 4, &dInferenceOutputSize[3], sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

        gpulz::sparseGpulzComp(dQuantizedOutput, dToWriteBack, gpulzCompedSize + threadIdx, dInferenceOutputSizeTotal, gpuIdx, stream);

        cudaStreamSynchronize(stream);

        uint8_t *hOutput;
        cudaMallocHost(&hOutput, gpulzCompedSize[threadIdx] + 5 * sizeof(uint32_t));
        cudaMemcpyAsync(hOutput, dToWriteBack, gpulzCompedSize[threadIdx] + 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        io::write_array_to_binary<uint8_t>(outputPathArr[threadIdx], hOutput, gpulzCompedSize[threadIdx] + 5 * sizeof(uint32_t));

        cudaFreeHost(hOutput);
        cudaFreeHost(hInputArr);

        cudaFree(dInputArrFloat);
        cudaFree(dQuantizedOutput);
        cudaFree(dToWriteBack);
    }
    cudaDeviceSynchronize();
    cudaFreeHost(myStream);

    auto end = std::chrono::high_resolution_clock::now();

    float compressionRatio[4];
    for (int i = 0; i < 4; i++)
    {
        compressionRatio[i] = (float)binaryFileSize / ((float)gpulzCompedSize[i] + 5.0f * sizeof(uint32_t));
        std::cout << "Compression ratio for subimage " << i << ": " << compressionRatio[i] << std::endl;
    }
    // std::cout << "Compression ratio: " << compressionRatio << std::endl;

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    return;
}

void ompDecompress(const std::string &modelPath, const std::string &inputPath, const std::string &outputPath)
{
    std::cout << "Decompressing..." << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int gpuIdx = 0;
    cudaSetDevice(gpuIdx);

    at::cuda::CUDAStream *myStream;
    cudaMallocHost(&myStream, 4 * sizeof(at::cuda::CUDAStream));

    for (int i = 0; i < 4; i++)
    {
        myStream[i] = at::cuda::getStreamFromPool(false, gpuIdx);
    }

    at::cuda::setCurrentCUDAStream(myStream[0]);

    const std::string deviceString = "cuda:" + std::to_string(gpuIdx);

    torch::jit::script::Module synthesisModel;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        synthesisModel = torch::jit::load(modelPath, deviceString);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }

    std::string inputPathArr[4];
    std::string outputPathArr[4];
    for (int i = 0; i < 4; i++)
    {
        inputPathArr[i] = inputPath + "_" + std::to_string(i) + ".comp";
        outputPathArr[i] = outputPath + "_" + std::to_string(i) + ".decomp";
    }

#pragma omp parallel num_threads(4)
    {
        int threadIdx = omp_get_thread_num();
        // printf("Thread %d\n", threadIdx);

        cudaStream_t stream = myStream[threadIdx].stream();

        uint32_t x = 0;
        uint32_t y = 0;

        uint32_t dInferenceOutputSize0 = 0;
        uint32_t dInferenceOutputSize1 = 0;
        uint32_t dInferenceOutputSize2 = 0;

        uint32_t *hInputArr;
        size_t binaryFileSize = io::fileSize(inputPathArr[threadIdx]);
        cudaMallocHost(&hInputArr, binaryFileSize);
        io::read_binary_to_array<uint32_t>(inputPathArr[threadIdx], hInputArr, binaryFileSize / sizeof(uint32_t));

        x = hInputArr[0];
        y = hInputArr[1];

        dInferenceOutputSize0 = hInputArr[2];
        dInferenceOutputSize1 = hInputArr[3];
        dInferenceOutputSize2 = hInputArr[4];

        uint8_t *dCompressedOutput;
        cudaMallocAsync(&dCompressedOutput, binaryFileSize - sizeof(uint32_t) * 5, stream);
        cudaMemcpyAsync(dCompressedOutput, hInputArr + 5, binaryFileSize - sizeof(uint32_t) * 5, cudaMemcpyHostToDevice, stream);

        INPUT_TYPE *dDecompedOutput;
        cudaMallocAsync(&dDecompedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(INPUT_TYPE), stream);

        gpulz::decompress(dCompressedOutput, dDecompedOutput, gpuIdx, stream);

        float *dDequantizedOutput;
        cudaMallocAsync(&dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 * sizeof(float), stream);

        // Kernel launch parameters
        int threadsPerBlock = 256;
        int blocksPerGrid = (dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2 + threadsPerBlock - 1) / threadsPerBlock;

        dequantize<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dDecompedOutput, dDequantizedOutput, dInferenceOutputSize0 * dInferenceOutputSize1 * dInferenceOutputSize2);

        cudaStreamSynchronize(stream);

        auto dims = torch::IntArrayRef{1, dInferenceOutputSize0, dInferenceOutputSize1, dInferenceOutputSize2};
        auto inputTensor = torch::from_blob(dDequantizedOutput, dims, deviceString);

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> synthesisModelInputs;
        synthesisModelInputs.push_back(inputTensor);

        // Execute the model and turn its output into a tensor.
        at::Tensor tmp_output = synthesisModel.forward(synthesisModelInputs).toTensor();
        cudaStreamSynchronize(stream);

        auto dSynthesisOutput = tmp_output.data_ptr<float>();

        float *hOutput;
        cudaMallocHost(&hOutput, 3 * x * y * sizeof(float));
        cudaMemcpyAsync(hOutput, dSynthesisOutput, 3 * x * y * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        io::write_array_to_binary<float>(outputPathArr[threadIdx], hOutput, 3 * x * y);

        cudaFreeHost(hInputArr);
        cudaFreeHost(hOutput);

        cudaFree(dCompressedOutput);
        cudaFree(dDecompedOutput);
        cudaFree(dDequantizedOutput);
    }

    cudaFreeHost(myStream);

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the time elapsed
    std::cout << "Time taken for operation: " << duration.count() << " microseconds." << std::endl;

    return;
}

void printHelp()
{
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -h        Display this help message and exit\n"
              << "  -m <arg>  the path to the model\n"
              << "  -c to launch compression\n"
              << "  -x to specify the x dimension if in compression mode\n"
              << "  -y to specify the y dimension if in compression mode\n"
              << "  -d to launch decompression\n"
              << "  -i <arg>  path to the input file\n"
              << "  -o <arg>  path to the output file\n"
              << "  -p to enable openmp multi thread pipeline\n"
              << "  -g to enable GDS I/O\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    std::string mValue, iValue, oValue;
    bool compressionEnabled = false;
    bool decompressionEnabled = false;
    bool helpRequested = false;
    bool parallelEnabled = false;
    bool gdsEnabled = false;
    int opt;
    uint32_t xValue, yValue;

    // ':' after an option indicates that it expects a value. No ':' means no value expected.
    while ((opt = getopt(argc, argv, "hm:cdi:o:x:y:pg")) != -1)
    {
        switch (opt)
        {
        case 'h':
            helpRequested = true;
            break;
        case 'm':
            mValue = optarg;
            break;
        case 'p':
            parallelEnabled = true;
            break;
        case 'c':
            compressionEnabled = true;
            break;
        case 'd':
            decompressionEnabled = true;
            break;
        case 'g':
            gdsEnabled = true;
            break;
        case 'i':
            iValue = optarg;
            break;
        case 'o':
            oValue = optarg;
            break;
        case 'x':
            xValue = static_cast<uint32_t>(std::stoul(optarg));
            break;
        case 'y':
            yValue = static_cast<uint32_t>(std::stoul(optarg));
            break;
        case '?': // Option not recognized
            // getopt automatically prints an error message, so we might want to return an error code
            return 1;
        default:
            // Unhandled options; this should not happen.
            break;
        }
    }

    // Check if help was requested. If so, display help and exit.
    if (helpRequested)
    {
        printHelp();
        return 0; // Exit after displaying help
    }

    if (compressionEnabled && decompressionEnabled)
    {
        std::cerr << "Error: cannot enable both compression and decompression at the same time" << std::endl;
        return 1;
    }
    if (!compressionEnabled && !decompressionEnabled)
    {
        std::cerr << "Error: must enable either compression or decompression" << std::endl;
        return 1;
    }
    if (compressionEnabled)
    {
        if (mValue.empty() || iValue.empty() || oValue.empty())
        {
            std::cerr << "Error: missing required arguments for compression" << std::endl;
            return 1;
        }
        if (!parallelEnabled && !gdsEnabled)
        {
            compress(mValue, iValue, oValue, xValue, yValue);
        }
        else if (gdsEnabled)
        {
            gdsCompress(mValue, iValue, oValue, xValue, yValue);
        }
        else if (parallelEnabled)
        {
            ompCompress(mValue, iValue, oValue, xValue, yValue);
        }
    }
    if (decompressionEnabled)
    {
        if (mValue.empty() || iValue.empty() || oValue.empty())
        {
            std::cerr << "Error: missing required arguments for decompression" << std::endl;
            return 1;
        }
        if (!parallelEnabled && !gdsEnabled)
        {
            decompress(mValue, iValue, oValue);
        }
        else if (gdsEnabled)
        {
            gdsDecompress(mValue, iValue, oValue);
        }
        else if (parallelEnabled)
        {
            ompDecompress(mValue, iValue, oValue);
        }
    }

    return 0;
}