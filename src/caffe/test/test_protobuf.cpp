// this is simply a script that serializing protocol buffer in text format.

#include <string>
#include <google/protobuf/text_format.h>
#include "gtest/gtest.h"
#include "caffe/proto/caffe.pb.h"

using namespace std;

namespace caffe {

class ProtoTest : public ::testing::Test {};

TEST_F(ProtoTest, TestSerialization) {
  LayerParameter param;
  param.set_name("test");
  param.set_type("dummy");
  cout << "Printing in binary format." << endl;
  cout << param.SerializeAsString() << endl;
  cout << "Printing in text format." << endl;
  string str;
  google::protobuf::TextFormat::PrintToString(param, &str);
  cout << str << endl;
  EXPECT_TRUE(true);
}

}  // namespce caffe
