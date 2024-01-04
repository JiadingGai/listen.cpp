#include <gtest/gtest.h>
#include <listen.h>
#include <string>
#define STRING(x) #x
#define XSTRING(x) STRING(x)

TEST(WhisperOperatorUnitTests, FirstConv1) {
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
  auto result = conv1d(input, conv1_weight, conv1_bias);

  /* std::cout << "conv1_bias = " << conv1_bias; */
  /* std::cout << "conv1_weight = " << conv1_weight; */
  const auto gold_conv1 = read_binary<float>(gold_conv1_fn, 384 * 300);
  float max_error = FLT_MIN;
  for (int i = 0; i < gold_conv1.size(); i++) {
    max_error = std::abs(gold_conv1[i] - result[i]) > max_error ? std::abs(gold_conv1[i] - result[i]) : max_error;
    /* std::cout << "(gold, result) = " << gold_conv1[i] << ", " << result[i] << "\n"; */
  }
  if (max_error >= 6e-6)
    std::cout << "(SimpleTheoryOfTypes) max_error = " <<max_error << std::endl;

  ASSERT_TRUE(max_error < 6e-6);
}

// The main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
