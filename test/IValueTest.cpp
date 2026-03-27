#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>
#include <torch/library.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/file_manager.h"

namespace c10 {
struct IValue;
}

namespace torch {
class IValue;
template <typename T>
class intrusive_ptr;
template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args);
}  // namespace torch

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

template <typename T, typename = void>
struct is_usable_ivalue : std::false_type {};

template <typename T>
struct is_usable_ivalue<
    T,
    std::void_t<decltype(T(true)),
                decltype(T(int64_t{1})),
                decltype(T(std::string("ivalue"))),
                decltype(std::declval<const T&>().template to<int64_t>())>>
    : std::true_type {};

using CompatIValue = std::conditional_t<is_usable_ivalue<c10::IValue>::value,
                                        c10::IValue,
                                        torch::IValue>;

static_assert(is_usable_ivalue<CompatIValue>::value,
              "No usable IValue type found in current backend.");

class IValueTestCustomHolder : public torch::CustomClassHolder {};

std::string to_lower_ascii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
  return value;
}

bool contains_token_ci(const std::string& text, const std::string& token) {
  return to_lower_ascii(text).find(to_lower_ascii(token)) != std::string::npos;
}

template <typename T, typename = void>
struct has_is_bool_snake : std::false_type {};
template <typename T>
struct has_is_bool_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_bool())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_is_bool_camel : std::false_type {};
template <typename T>
struct has_is_bool_camel<
    T,
    std::void_t<decltype(std::declval<const T&>().isBool())>> : std::true_type {
};

template <typename T>
bool iv_is_bool(const T& iv) {
  if constexpr (has_is_bool_snake<T>::value) {
    return iv.is_bool();
  } else {
    return iv.isBool();
  }
}

template <typename T, typename = void>
struct has_is_int_snake : std::false_type {};
template <typename T>
struct has_is_int_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_int())>> : std::true_type {
};

template <typename T, typename = void>
struct has_is_int_camel : std::false_type {};
template <typename T>
struct has_is_int_camel<T,
                        std::void_t<decltype(std::declval<const T&>().isInt())>>
    : std::true_type {};

template <typename T>
bool iv_is_int(const T& iv) {
  if constexpr (has_is_int_snake<T>::value) {
    return iv.is_int();
  } else {
    return iv.isInt();
  }
}

template <typename T, typename = void>
struct has_is_double_snake : std::false_type {};
template <typename T>
struct has_is_double_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_double())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_is_double_camel : std::false_type {};
template <typename T>
struct has_is_double_camel<
    T,
    std::void_t<decltype(std::declval<const T&>().isDouble())>>
    : std::true_type {};

template <typename T>
bool iv_is_double(const T& iv) {
  if constexpr (has_is_double_snake<T>::value) {
    return iv.is_double();
  } else {
    return iv.isDouble();
  }
}

template <typename T, typename = void>
struct has_is_string_snake : std::false_type {};
template <typename T>
struct has_is_string_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_string())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_is_string_camel : std::false_type {};
template <typename T>
struct has_is_string_camel<
    T,
    std::void_t<decltype(std::declval<const T&>().isString())>>
    : std::true_type {};

template <typename T>
bool iv_is_string(const T& iv) {
  if constexpr (has_is_string_snake<T>::value) {
    return iv.is_string();
  } else {
    return iv.isString();
  }
}

template <typename T, typename = void>
struct has_is_list_snake : std::false_type {};
template <typename T>
struct has_is_list_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_list())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_is_list_camel : std::false_type {};
template <typename T>
struct has_is_list_camel<
    T,
    std::void_t<decltype(std::declval<const T&>().isList())>> : std::true_type {
};

template <typename T>
bool iv_is_list(const T& iv) {
  if constexpr (has_is_list_snake<T>::value) {
    return iv.is_list();
  } else {
    return iv.isList();
  }
}

template <typename T, typename = void>
struct has_is_tuple_snake : std::false_type {};
template <typename T>
struct has_is_tuple_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_tuple())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_is_tuple_camel : std::false_type {};
template <typename T>
struct has_is_tuple_camel<
    T,
    std::void_t<decltype(std::declval<const T&>().isTuple())>>
    : std::true_type {};

template <typename T>
bool iv_is_tuple(const T& iv) {
  if constexpr (has_is_tuple_snake<T>::value) {
    return iv.is_tuple();
  } else {
    return iv.isTuple();
  }
}

template <typename T, typename = void>
struct has_is_custom_class_snake : std::false_type {};
template <typename T>
struct has_is_custom_class_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().is_custom_class())>>
    : std::true_type {};

template <typename T>
bool iv_is_custom_class(const T& iv) {
  if constexpr (has_is_custom_class_snake<T>::value) {
    return iv.is_custom_class();
  } else {
    return iv.isCustomClass();
  }
}

template <typename T, typename = void>
struct has_to_bool_snake : std::false_type {};
template <typename T>
struct has_to_bool_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_bool())>>
    : std::true_type {};

template <typename T>
bool iv_to_bool(const T& iv) {
  if constexpr (has_to_bool_snake<T>::value) {
    return iv.to_bool();
  } else {
    return iv.toBool();
  }
}

template <typename T, typename = void>
struct has_to_int_snake : std::false_type {};
template <typename T>
struct has_to_int_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_int())>> : std::true_type {
};

template <typename T>
int64_t iv_to_int(const T& iv) {
  if constexpr (has_to_int_snake<T>::value) {
    return iv.to_int();
  } else {
    return iv.toInt();
  }
}

template <typename T, typename = void>
struct has_to_double_snake : std::false_type {};
template <typename T>
struct has_to_double_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_double())>>
    : std::true_type {};

template <typename T>
double iv_to_double(const T& iv) {
  if constexpr (has_to_double_snake<T>::value) {
    return iv.to_double();
  } else {
    return iv.toDouble();
  }
}

template <typename T, typename = void>
struct has_to_string_snake : std::false_type {};
template <typename T>
struct has_to_string_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_string())>>
    : std::true_type {};

template <typename T>
std::string iv_to_string(const T& iv) {
  if constexpr (has_to_string_snake<T>::value) {
    return iv.to_string();
  } else {
    return iv.toStringRef();
  }
}

template <typename T, typename = void>
struct has_to_string_view_snake : std::false_type {};
template <typename T>
struct has_to_string_view_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_string_view())>>
    : std::true_type {};

template <typename T>
std::string_view iv_to_string_view(const T& iv) {
  if constexpr (has_to_string_view_snake<T>::value) {
    return iv.to_string_view();
  } else {
    return iv.toStringView();
  }
}

template <typename T, typename = void>
struct has_to_tensor_snake : std::false_type {};
template <typename T>
struct has_to_tensor_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_tensor())>>
    : std::true_type {};

template <typename T>
at::Tensor iv_to_tensor(const T& iv) {
  if constexpr (has_to_tensor_snake<T>::value) {
    return iv.to_tensor();
  } else {
    return iv.toTensor();
  }
}

template <typename T, typename = void>
struct has_to_scalar_type_snake : std::false_type {};
template <typename T>
struct has_to_scalar_type_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_scalar_type())>>
    : std::true_type {};

template <typename T>
at::ScalarType iv_to_scalar_type(const T& iv) {
  if constexpr (has_to_scalar_type_snake<T>::value) {
    return iv.to_scalar_type();
  } else {
    return iv.toScalarType();
  }
}

template <typename T, typename = void>
struct has_to_list_snake : std::false_type {};
template <typename T>
struct has_to_list_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_list())>>
    : std::true_type {};

template <typename T>
auto iv_to_list(const T& iv) {
  if constexpr (has_to_list_snake<T>::value) {
    return iv.to_list();
  } else {
    return iv.toList();
  }
}

template <typename ListT, typename = void>
struct has_get_index : std::false_type {};
template <typename ListT>
struct has_get_index<ListT,
                     std::void_t<decltype(std::declval<const ListT&>().get(0))>>
    : std::true_type {};

template <typename ListT>
CompatIValue list_get(const ListT& list, size_t idx) {
  if constexpr (has_get_index<ListT>::value) {
    return list.get(idx);
  } else {
    return list[idx];
  }
}

template <typename T, typename = void>
struct has_to_tuple_snake : std::false_type {};
template <typename T>
struct has_to_tuple_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().to_tuple())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_to_tuple_ref_camel : std::false_type {};
template <typename T>
struct has_to_tuple_ref_camel<
    T,
    std::void_t<
        decltype(std::declval<const T&>().toTupleRef().elements().size())>>
    : std::true_type {};

template <typename T>
int64_t iv_tuple_size(const T& iv) {
  if constexpr (has_to_tuple_snake<T>::value) {
    return static_cast<int64_t>(iv.to_tuple().size());
  } else {
    return static_cast<int64_t>(iv.toTupleRef().elements().size());
  }
}

template <typename T, typename ClassT, typename = void>
struct has_to_custom_class_snake : std::false_type {};
template <typename T, typename ClassT>
struct has_to_custom_class_snake<
    T,
    ClassT,
    std::void_t<
        decltype(std::declval<const T&>().template to_custom_class<ClassT>())>>
    : std::true_type {};

template <typename ClassT, typename T>
auto iv_to_custom_class(const T& iv) {
  if constexpr (has_to_custom_class_snake<T, ClassT>::value) {
    return iv.template to_custom_class<ClassT>();
  } else {
    return iv.template toCustomClass<ClassT>();
  }
}

template <typename T, typename = void>
struct has_try_to_bool : std::false_type {};
template <typename T>
struct has_try_to_bool<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_bool(
        std::declval<bool&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_bool(const T& iv, bool* out) {
  if constexpr (has_try_to_bool<T>::value) {
    return iv.try_to_bool(*out);
  }
  if (iv_is_bool(iv)) {
    *out = iv_to_bool(iv);
    return true;
  }
  if (iv_is_int(iv)) {
    *out = iv_to_int(iv) != 0;
    return true;
  }
  if (iv_is_double(iv)) {
    *out = iv_to_double(iv) != 0.0;
    return true;
  }
  return false;
}

template <typename T, typename = void>
struct has_try_to_int : std::false_type {};
template <typename T>
struct has_try_to_int<T,
                      std::void_t<decltype(std::declval<const T&>().try_to_int(
                          std::declval<int&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_int(const T& iv, int* out) {
  if constexpr (has_try_to_int<T>::value) {
    return iv.try_to_int(*out);
  }
  if (iv_is_int(iv)) {
    *out = static_cast<int>(iv_to_int(iv));
    return true;
  }
  if (iv_is_double(iv)) {
    *out = static_cast<int>(iv_to_double(iv));
    return true;
  }
  return false;
}

template <typename T, typename = void>
struct has_try_to_double : std::false_type {};
template <typename T>
struct has_try_to_double<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_double(
        std::declval<double&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_double(const T& iv, double* out) {
  if constexpr (has_try_to_double<T>::value) {
    return iv.try_to_double(*out);
  }
  if (iv_is_double(iv)) {
    *out = iv_to_double(iv);
    return true;
  }
  if (iv_is_int(iv)) {
    *out = static_cast<double>(iv_to_int(iv));
    return true;
  }
  return false;
}

template <typename T, typename = void>
struct has_try_to_string : std::false_type {};
template <typename T>
struct has_try_to_string<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_string(
        std::declval<std::string&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_string(const T& iv, std::string* out) {
  if constexpr (has_try_to_string<T>::value) {
    return iv.try_to_string(*out);
  }
  if (iv_is_string(iv)) {
    *out = iv_to_string(iv);
    return true;
  }
  return false;
}

template <typename T, typename = void>
struct has_try_to_tensor : std::false_type {};
template <typename T>
struct has_try_to_tensor<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_tensor(
        std::declval<at::Tensor&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_tensor(const T& iv, at::Tensor* out) {
  if constexpr (has_try_to_tensor<T>::value) {
    return iv.try_to_tensor(*out);
  }
  try {
    *out = iv_to_tensor(iv);
    return true;
  } catch (...) {
    return false;
  }
}

template <typename T, typename = void>
struct has_try_to_scalar_type : std::false_type {};
template <typename T>
struct has_try_to_scalar_type<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_scalar_type(
        std::declval<at::ScalarType&>()))>> : std::true_type {};

template <typename T>
bool iv_try_to_scalar_type(const T& iv, at::ScalarType* out) {
  if constexpr (has_try_to_scalar_type<T>::value) {
    return iv.try_to_scalar_type(*out);
  }
  if (iv_is_int(iv)) {
    *out = static_cast<at::ScalarType>(iv_to_int(iv));
    return true;
  }
  return false;
}

template <typename T, typename ValueT, typename = void>
struct has_try_to_optional_type_snake : std::false_type {};
template <typename T, typename ValueT>
struct has_try_to_optional_type_snake<
    T,
    ValueT,
    std::void_t<
        decltype(std::declval<const T&>().template try_to_optional_type<ValueT>(
            std::declval<std::optional<ValueT>&>()))>> : std::true_type {};

template <typename T, typename ValueT, typename = void>
struct has_to_optional_camel : std::false_type {};
template <typename T, typename ValueT>
struct has_to_optional_camel<
    T,
    ValueT,
    std::void_t<
        decltype(std::declval<const T&>().template toOptional<ValueT>())>>
    : std::true_type {};

template <typename ValueT, typename T>
bool iv_try_to_optional_type(const T& iv, std::optional<ValueT>* out) {
  if constexpr (has_try_to_optional_type_snake<T, ValueT>::value) {
    return iv.template try_to_optional_type<ValueT>(*out);
  } else if constexpr (has_to_optional_camel<T, ValueT>::value) {
    *out = iv.template toOptional<ValueT>();
    return true;
  } else {
    return false;
  }
}

template <typename T, typename ValueT, typename = void>
struct has_try_convert_to : std::false_type {};
template <typename T, typename ValueT>
struct has_try_convert_to<
    T,
    ValueT,
    std::void_t<
        decltype(std::declval<const T&>().template try_convert_to<ValueT>(
            std::declval<ValueT&>()))>> : std::true_type {};

template <typename ValueT, typename T>
bool iv_try_convert_to(const T& iv, ValueT* out) {
  if constexpr (has_try_convert_to<T, ValueT>::value) {
    return iv.template try_convert_to<ValueT>(*out);
  }
  try {
    *out = iv.template to<ValueT>();
    return true;
  } catch (...) {
    return false;
  }
}

template <typename T, typename = void>
struct has_try_to_custom_class_snake : std::false_type {};
template <typename T>
struct has_try_to_custom_class_snake<
    T,
    std::void_t<decltype(std::declval<const T&>().try_to_custom_class(
        std::declval<std::shared_ptr<torch::CustomClassHolder>&>(),
        std::declval<const std::string&>()))>> : std::true_type {};

template <typename T, typename ClassT, typename = void>
struct has_to_custom_class_camel : std::false_type {};
template <typename T, typename ClassT>
struct has_to_custom_class_camel<
    T,
    ClassT,
    std::void_t<
        decltype(std::declval<const T&>().template toCustomClass<ClassT>())>>
    : std::true_type {};

template <typename ClassT, typename T>
bool iv_try_to_custom_class(const T& iv,
                            const std::string& expected_class_name) {
  if constexpr (has_try_to_custom_class_snake<T>::value) {
    std::shared_ptr<torch::CustomClassHolder> out;
    return iv.try_to_custom_class(out, expected_class_name);
  } else if constexpr (has_to_custom_class_camel<T, ClassT>::value) {
    try {
      auto out = iv.template toCustomClass<ClassT>();
      return static_cast<bool>(out);
    } catch (...) {
      return false;
    }
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct has_get_custom_class_name : std::false_type {};
template <typename T>
struct has_get_custom_class_name<
    T,
    std::void_t<decltype(std::declval<const T&>().get_custom_class_name())>>
    : std::true_type {};

template <typename T>
std::string iv_get_custom_class_name(const T& iv) {
  if constexpr (has_get_custom_class_name<T>::value) {
    return iv.get_custom_class_name();
  } else {
    if (!iv.isCustomClass()) {
      throw std::runtime_error("Not a custom class");
    }
    return iv.type()->repr_str();
  }
}

template <typename T, typename = void>
struct has_type_string : std::false_type {};
template <typename T>
struct has_type_string<
    T,
    std::void_t<decltype(std::declval<const T&>().type_string())>>
    : std::true_type {};

template <typename T>
std::string iv_type_string(const T& iv) {
  if constexpr (has_type_string<T>::value) {
    return iv.type_string();
  } else {
    return iv.type()->repr_str();
  }
}

template <typename T, typename = void>
struct has_to_repr : std::false_type {};
template <typename T>
struct has_to_repr<T, std::void_t<decltype(std::declval<const T&>().to_repr())>>
    : std::true_type {};

template <typename T>
std::string iv_to_repr(const T& iv) {
  if constexpr (has_to_repr<T>::value) {
    return iv.to_repr();
  } else {
    std::ostringstream os;
    iv.repr(os, [](std::ostream&, const c10::IValue&) { return false; });
    return os.str();
  }
}

class IValueTest : public ::testing::Test {};

TEST_F(IValueTest, CoreIsToMethods) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "CoreIsToMethods ";

  CompatIValue iv_bool(true);
  CompatIValue iv_int(static_cast<int64_t>(42));
  CompatIValue iv_double(3.5);
  CompatIValue iv_string(std::string("hello_ivalue"));
  CompatIValue iv_tensor(
      at::zeros({2, 3}, at::TensorOptions().dtype(at::kFloat)));
  CompatIValue iv_scalar_type(at::kDouble);

  file << std::to_string(iv_is_bool(iv_bool) ? 1 : 0) << " ";
  file << std::to_string(iv_to_bool(iv_bool) ? 1 : 0) << " ";
  file << std::to_string(iv_to_int(iv_int)) << " ";
  file << std::to_string(iv_to_double(iv_double)) << " ";
  file << iv_to_string(iv_string) << " ";
  file << std::string(iv_to_string_view(iv_string)) << " ";
  file << std::to_string(iv_to_tensor(iv_tensor).numel()) << " ";
  file << std::to_string(static_cast<int>(iv_to_scalar_type(iv_scalar_type)))
       << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IValueTest, ListAndTupleMethods) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ListAndTupleMethods ";

  CompatIValue iv_list(std::vector<int64_t>{1, 2, 3, 4});
  CompatIValue iv_tuple(
      std::tuple<int64_t, double, std::string>{7, 2.25, "tuple_v"});

  auto list_obj = iv_to_list(iv_list);
  auto list_first = list_get(list_obj, 0);
  auto tuple_std = iv_tuple.to<std::tuple<int64_t, double, std::string>>();

  file << std::to_string(iv_is_list(iv_list) ? 1 : 0) << " ";
  file << std::to_string(static_cast<int64_t>(list_obj.size())) << " ";
  file << std::to_string(iv_to_int(list_first)) << " ";
  file << std::to_string(iv_is_tuple(iv_tuple) ? 1 : 0) << " ";
  file << std::to_string(iv_tuple_size(iv_tuple)) << " ";
  file << std::to_string(std::get<0>(tuple_std)) << " ";
  file << std::to_string(std::get<1>(tuple_std)) << " ";
  file << std::get<2>(tuple_std) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IValueTest, TryToMethods) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TryToMethods ";

  CompatIValue iv_from_int(static_cast<int64_t>(9));
  CompatIValue iv_from_double(6.75);
  CompatIValue iv_from_string(std::string("try_string"));
  CompatIValue iv_from_tensor(
      at::zeros({1, 1, 1}, at::TensorOptions().dtype(at::kDouble)));
  CompatIValue iv_from_scalar_type(at::kFloat);

  bool out_bool = false;
  int out_int = 0;
  double out_double = 0.0;
  std::string out_string;
  at::Tensor out_tensor;
  at::ScalarType out_scalar_type = at::kFloat;

  bool ok_bool = iv_try_to_bool(iv_from_int, &out_bool);
  bool ok_int = iv_try_to_int(iv_from_double, &out_int);
  bool ok_double = iv_try_to_double(iv_from_int, &out_double);
  bool ok_string = iv_try_to_string(iv_from_string, &out_string);
  bool ok_tensor = iv_try_to_tensor(iv_from_tensor, &out_tensor);
  bool ok_scalar = iv_try_to_scalar_type(iv_from_scalar_type, &out_scalar_type);

  file << std::to_string(ok_bool ? 1 : 0) << " ";
  file << std::to_string(out_bool ? 1 : 0) << " ";
  file << std::to_string(ok_int ? 1 : 0) << " ";
  file << std::to_string(out_int) << " ";
  file << std::to_string(ok_double ? 1 : 0) << " ";
  file << std::to_string(out_double) << " ";
  file << std::to_string(ok_string ? 1 : 0) << " ";
  file << out_string << " ";
  file << std::to_string(ok_tensor ? 1 : 0) << " ";
  file << std::to_string(ok_tensor ? out_tensor.numel() : -1) << " ";
  file << std::to_string(ok_scalar ? 1 : 0) << " ";
  file << std::to_string(static_cast<int>(out_scalar_type)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IValueTest, ToTemplateExtended) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToTemplateExtended ";

  CompatIValue iv_string(std::string("generic_to_template"));
  CompatIValue iv_vector(std::vector<int64_t>{9, 8, 7});
  CompatIValue iv_tuple(std::tuple<int64_t, double, std::string>{4, 1.5, "x"});
  CompatIValue iv_optional(std::optional<int64_t>{33});

  auto out_view = iv_string.to<std::string_view>();
  auto out_vector = iv_vector.to<std::vector<int64_t>>();
  auto out_tuple = iv_tuple.to<std::tuple<int64_t, double, std::string>>();
  auto out_optional = iv_optional.to<std::optional<int64_t>>();

  file << std::string(out_view) << " ";
  file << std::to_string(static_cast<int64_t>(out_vector.size())) << " ";
  file << std::to_string(out_vector[0]) << " ";
  file << std::to_string(std::get<0>(out_tuple)) << " ";
  file << std::to_string(std::get<1>(out_tuple)) << " ";
  file << std::get<2>(out_tuple) << " ";
  file << std::to_string(out_optional.has_value() ? 1 : 0) << " ";
  file << std::to_string(out_optional.value_or(-1)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IValueTest, OptionalAndConvertMethods) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OptionalAndConvertMethods ";

  CompatIValue iv_none;
  CompatIValue iv_bool(true);
  CompatIValue iv_int(static_cast<int64_t>(21));
  CompatIValue iv_string(std::string("convert_me"));
  CompatIValue iv_vector(std::vector<int64_t>{5, 6, 7});
  CompatIValue iv_tuple(
      std::tuple<int64_t, double, std::string>{8, 4.5, "cvt"});

  std::optional<int64_t> out_none_optional;
  std::optional<int64_t> out_some_optional;
  bool out_bool = false;
  std::string out_string;
  std::vector<int64_t> out_vector;
  std::tuple<int64_t, double, std::string> out_tuple;

  bool ok_none_optional = iv_try_to_optional_type(iv_none, &out_none_optional);
  bool ok_some_optional = iv_try_to_optional_type(iv_int, &out_some_optional);
  bool ok_bool = iv_try_convert_to(iv_bool, &out_bool);
  bool ok_string = iv_try_convert_to(iv_string, &out_string);
  bool ok_vector = iv_try_convert_to(iv_vector, &out_vector);
  bool ok_tuple = iv_try_convert_to(iv_tuple, &out_tuple);

  file << std::to_string(ok_none_optional ? 1 : 0) << " ";
  file << std::to_string(out_none_optional.has_value() ? 1 : 0) << " ";
  file << std::to_string(ok_some_optional ? 1 : 0) << " ";
  file << std::to_string(out_some_optional.value_or(-1)) << " ";
  file << std::to_string(ok_bool ? 1 : 0) << " ";
  file << std::to_string(out_bool ? 1 : 0) << " ";
  file << std::to_string(ok_string ? 1 : 0) << " ";
  file << out_string << " ";
  file << std::to_string(ok_vector ? 1 : 0) << " ";
  file << std::to_string(static_cast<int64_t>(out_vector.size())) << " ";
  file << std::to_string(ok_tuple ? 1 : 0) << " ";
  file << std::to_string(std::get<0>(out_tuple)) << " ";
  file << std::to_string(std::get<1>(out_tuple)) << " ";
  file << std::get<2>(out_tuple) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IValueTest, ReprAndTypeStringMethods) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReprAndTypeStringMethods ";

  CompatIValue iv_int(static_cast<int64_t>(123));
  CompatIValue iv_string(std::string("repr_value"));
  CompatIValue iv_list(std::vector<int64_t>{1, 2, 3});
  CompatIValue iv_tuple(
      std::tuple<int64_t, double, std::string>{2, 7.5, "tuple_repr"});

  std::string int_type = iv_type_string(iv_int);
  std::string list_type = iv_type_string(iv_list);
  std::string tuple_type = iv_type_string(iv_tuple);

  std::string int_repr = iv_to_repr(iv_int);
  std::string string_repr = iv_to_repr(iv_string);
  std::string list_repr = iv_to_repr(iv_list);
  std::string tuple_repr = iv_to_repr(iv_tuple);
  bool custom_to_failed = false;
  bool custom_try_ok = iv_try_to_custom_class<IValueTestCustomHolder>(
      iv_int, "IValueTestCustomHolder");
  bool custom_name_failed = false;

  try {
    (void)iv_to_custom_class<IValueTestCustomHolder>(iv_int);
  } catch (...) {
    custom_to_failed = true;
  }

  try {
    (void)iv_get_custom_class_name(iv_int);
  } catch (...) {
    custom_name_failed = true;
  }

  file << std::to_string(iv_is_custom_class(iv_int) ? 1 : 0) << " ";
  file << std::to_string(custom_to_failed ? 1 : 0) << " ";
  file << std::to_string(custom_try_ok ? 1 : 0) << " ";
  file << std::to_string(custom_name_failed ? 1 : 0) << " ";
  file << std::to_string(contains_token_ci(int_type, "int") ? 1 : 0) << " ";
  file << std::to_string(contains_token_ci(list_type, "list") ? 1 : 0) << " ";
  file << std::to_string(contains_token_ci(tuple_type, "tuple") ? 1 : 0) << " ";
  file << std::to_string(contains_token_ci(int_repr, "123") ? 1 : 0) << " ";
  file << std::to_string(contains_token_ci(string_repr, "repr_value") ? 1 : 0)
       << " ";
  file << std::to_string(list_repr.find('[') != std::string::npos ? 1 : 0)
       << " ";
  file << std::to_string(tuple_repr.find('(') != std::string::npos ||
                                 contains_token_ci(tuple_repr, "tuple")
                             ? 1
                             : 0)
       << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
