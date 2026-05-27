from std.python import Python, PythonObject

# TODO: Investigate https://github.com/ehsanmok/flare.


struct LazyLoader:
    var cache_dir: String

    def __init__(out self, cache_dir: String = "~/.mograd/datasets"):
        self.cache_dir = cache_dir

    def ensure_cached(self, url: String, filename: String) raises -> String:
        var os = Python.import_module("os")
        var urllib = Python.import_module("urllib.request")

        var expanded = String(os.path.expanduser(self.cache_dir))
        _ = os.makedirs(expanded, exist_ok=True)
        var dest = expanded + "/" + filename

        if not Bool(os.path.exists(dest)):
            print("Downloading", filename, "...")
            _ = urllib.urlretrieve(url, dest)
            print("Saved to", dest)

        return dest

    def ensure_cached_idx(
        self, url: String, gz_name: String, out_name: String, is_images: Bool
    ) raises -> String:
        var os = Python.import_module("os")
        var expanded = String(os.path.expanduser(self.cache_dir))
        var out_path = expanded + "/" + out_name

        if Bool(os.path.exists(out_path)):
            return out_path

        var gz_path = self.ensure_cached(url, gz_name)
        self._convert_idx(gz_path, out_path, is_images)
        return out_path

    def _convert_idx(
        self, gz_path: String, out_path: String, is_images: Bool
    ) raises:
        var gzip = Python.import_module("gzip")
        var np = Python.import_module("numpy")

        var f = gzip.open(gz_path, "rb")
        var raw = f.read()
        _ = f.close()

        if is_images:
            # IDX3: 16-byte header (magic, count, rows, cols), then uint8 pixels
            var arr = np.frombuffer(raw, dtype=np.uint8, offset=16)
            arr = arr.reshape(-1, 784).astype(np.float32) / 255.0
            arr.tofile(out_path)
        else:
            # IDX1: 8-byte header (magic, count), then uint8 labels
            var arr = np.frombuffer(raw, dtype=np.uint8, offset=8)
            arr = arr.astype(np.float32)
            arr.tofile(out_path)
