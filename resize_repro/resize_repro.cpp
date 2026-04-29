#include <ATen/ATen.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

std::string test_resize(at::Tensor t) {
  TORCH_CHECK(t.dtype() == at::kInt, "expected int32 tensor");
  TORCH_CHECK(t.numel() == 2, "expected 2-element tensor");

  std::string log;
  log += "before resize_: numel=" + std::to_string(t.numel()) +
         " storage_bytes=" + std::to_string(t.storage().nbytes()) + "\n";

  t.resize_({4});

  const size_t bytes_after = t.storage().nbytes();
  log += "after resize_: numel=" + std::to_string(t.numel()) +
         " storage_bytes=" + std::to_string(bytes_after) + "\n";

  if (bytes_after < 16) {
    throw std::runtime_error(
        log + "BUG: storage().nbytes()=" + std::to_string(bytes_after) +
        " expected at least 16 -- SyncStorageFromTensor rebuilt from stale "
        "PlainAlloc (8 B) instead of keeping the updated impl.");
  }

  log += "PASS\n";
  return log;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_resize",
        &test_resize,
        "Reproduce resize_() storage().nbytes() bug");
}
