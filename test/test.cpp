#include <gtest/gtest.h>
#include <listen.h>
#include <string>
#define STRING(x) #x
#define XSTRING(x) STRING(x)

TEST(WhisperOperatorUnitTests, Conv1GeluConv2Gelu) {
  // whisper source root is passed from cmake; see add_compile_definition in CMakeLists.txt
  /* std::cout << XSTRING(WHISPER_SOURCE_ROOT) << '\n'; */
  std::string whisper_home = XSTRING(WHISPER_SOURCE_ROOT);

  // read the weight and bias of conv1
  std::string weight_dir= whisper_home + "/test/model_weights/";
  std::string gold_input_fn = weight_dir + "tiny_en/gold_input.gaibin";
  std::string conv1_weight_fn = weight_dir + "tiny_en/encoder_conv1_weight.gaibin";
  std::string conv1_bias_fn = weight_dir + "tiny_en/encoder_conv1_bias.gaibin";
  std::string conv2_weight_fn = weight_dir + "tiny_en/encoder_conv2_weight.gaibin";
  std::string conv2_bias_fn = weight_dir + "tiny_en/encoder_conv2_bias.gaibin";
  // Golden's are generated from openai's reference impl like below:
  // 166         x = F.gelu(self.conv1(x))
  // 167         x.numpy().astype('single').flatten().tofile('__gold_1stconv1gelu.gaibin')
  std::string gold_output_fn = "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/gold_conv2gelu.gaibin";
  //std::string gold_output_fn = "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/gold_conv2.gaibin";
  std::map<std::string, int64_t> modelInfo {
    {conv1_weight_fn, 384 * 80 * 3},
    {conv1_bias_fn, 384},
    {conv2_weight_fn, 384 * 384 * 3},
    {conv2_bias_fn, 384},
  };

  const auto input = read_binary<float>(gold_input_fn, 1 * 80 * 3000);

  // Start profiling
  ProfilerStart("__profile_output.prof");

  // conv1 + relu
  const auto conv1_bias = read_binary<float>(conv1_bias_fn, 384);
  const auto conv1_weight = read_binary<float>(conv1_weight_fn, 384 * 80 * 3);
  const auto conv1_out = conv1d(input, {1, 80, 3000}, conv1_weight, {80, 384, 3}, conv1_bias);
  const auto tmp0 = gelu(conv1_out);

  // conv2 + relu
  const auto conv2_bias = read_binary<float>(conv2_bias_fn, 384);
  const auto conv2_weight = read_binary<float>(conv2_weight_fn, 384 * 384 * 3);
  const auto tmp1 = conv1d(tmp0, {1, 384, 3000}, conv2_weight, {384, 384, 3}, conv2_bias, /*stride=*/2);
  const auto result = gelu(tmp1);

  // Stop profiling
  ProfilerStop();

  const auto gold_conv2 = read_binary<float>(gold_output_fn, 384 * 1500);
  float max_error = FLT_MIN;
  for (int i = 0; i < gold_conv2.size(); i++) {
    max_error = std::abs(gold_conv2[i] - result[i]) > max_error ? std::abs(gold_conv2[i] - result[i]) : max_error;
    /* std::cout << "(gold, result) = " << gold_conv1[i] << ", " << result[i] << "\n"; */
  }

  const float atol = 5e-4;
  if (max_error >= atol)
    std::cout << "(SimpleTheoryOfTypes) max_error = " <<max_error << std::endl;

  ASSERT_TRUE(max_error < atol);
}

TEST(WhisperOperatorUnitTests, Conv1Gelu) {
  // whisper source root is passed from cmake; see add_compile_definition in CMakeLists.txt
  /* std::cout << XSTRING(WHISPER_SOURCE_ROOT) << '\n'; */
  std::string whisper_home = XSTRING(WHISPER_SOURCE_ROOT);

  // read the weight and bias of conv1
  std::string weight_dir= whisper_home + "/test/model_weights/";
  std::string gold_input_fn = weight_dir + "tiny_en/gold_input.gaibin";
  std::string conv1_weight_fn = weight_dir + "tiny_en/encoder_conv1_weight.gaibin";
  std::string conv1_bias_fn = weight_dir + "tiny_en/encoder_conv1_bias.gaibin";
  // Golden's are generated from openai's reference impl like below:
  // 166         x = F.gelu(self.conv1(x))
  // 167         x.numpy().astype('single').flatten().tofile('__gold_1stconv1gelu.gaibin')
  std::string gold_1stconv1gelu_fn = "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/gold_1stconv1gelu.gaibin";
  std::map<std::string, int64_t> modelInfo {
    {conv1_weight_fn, 384 * 80 * 3},
    {conv1_bias_fn, 384},
  };

  const auto input = read_binary<float>(gold_input_fn, 1 * 80 * 3000);
  const auto conv1_bias = read_binary<float>(conv1_bias_fn, 384);
  const auto conv1_weight = read_binary<float>(conv1_weight_fn, 384 * 80 * 3);
  const auto conv1_out = conv1d(input, {1, 80, 3000}, conv1_weight, {80, 384, 3}, conv1_bias);
  const auto result = gelu(conv1_out);

  /* std::cout << "conv1_bias = " << conv1_bias; */
  /* std::cout << "conv1_weight = " << conv1_weight; */
  const auto gold_conv1 = read_binary<float>(gold_1stconv1gelu_fn, 384 * 3000);
  float max_error = FLT_MIN;
  for (int i = 0; i < gold_conv1.size(); i++) {
    max_error = std::abs(gold_conv1[i] - result[i]) > max_error ? std::abs(gold_conv1[i] - result[i]) : max_error;
    /* std::cout << "(gold, result) = " << gold_conv1[i] << ", " << result[i] << "\n"; */
  }

  const float atol = 5e-4;
  if (max_error >= atol)
    std::cout << "(SimpleTheoryOfTypes) max_error = " <<max_error << std::endl;

  ASSERT_TRUE(max_error < atol);
}

TEST(WhisperOperatorUnitTests, Conv1) {
  // whisper source root is passed from cmake; see add_compile_definition in CMakeLists.txt
  /* std::cout << XSTRING(WHISPER_SOURCE_ROOT) << '\n'; */
  std::string whisper_home = XSTRING(WHISPER_SOURCE_ROOT);

  // read the weight and bias of conv1
  std::string weight_dir= whisper_home + "/test/model_weights/";
  std::string gold_input_fn = weight_dir + "tiny_en/gold_input.gaibin";
  std::string conv1_weight_fn = weight_dir + "tiny_en/encoder_conv1_weight.gaibin";
  std::string conv1_bias_fn = weight_dir + "tiny_en/encoder_conv1_bias.gaibin";
  std::string gold_conv1_fn = "/Users/jiadinggai/dev/WHISPER/gaiwhisper/test/gold_conv1.bin";
  std::map<std::string, int64_t> modelInfo {
    {conv1_weight_fn, 384 * 80 * 3},
    {conv1_bias_fn, 384},
  };

  const auto input = read_binary<float>(gold_input_fn, 1 * 80 * 3000);
  const auto conv1_bias = read_binary<float>(conv1_bias_fn, 384);
  const auto conv1_weight = read_binary<float>(conv1_weight_fn, 384 * 80 * 3);
  auto result = conv1d(input, {1, 80, 3000}, conv1_weight, {80, 384, 3}, conv1_bias);

  /* std::cout << "conv1_bias = " << conv1_bias; */
  /* std::cout << "conv1_weight = " << conv1_weight; */
  const auto gold_conv1 = read_binary<float>(gold_conv1_fn, 384 * 3000);
  float max_error = FLT_MIN;
  for (int i = 0; i < gold_conv1.size(); i++) {
    max_error = std::abs(gold_conv1[i] - result[i]) > max_error ? std::abs(gold_conv1[i] - result[i]) : max_error;
    /* std::cout << "(gold, result) = " << gold_conv1[i] << ", " << result[i] << "\n"; */
  }
  const float atol = 1e-5;
  if (max_error >= atol)
    std::cout << "(SimpleTheoryOfTypes) max_error = " <<max_error << std::endl;

  ASSERT_TRUE(max_error < atol);
}

// The main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
