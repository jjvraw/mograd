from std.collections import Set
from std.os import makedirs, remove
from std.os.path import exists, expanduser, getsize, join
from std.pathlib.path import Path
from std.python import Python, PythonObject


struct LazyLoader:
    var cache_dir: String

    def __init__(out self, cache_dir: String = "~/.mograd/datasets") raises:
        self.cache_dir = expanduser(cache_dir)
        makedirs(self.cache_dir, exist_ok=True)

    def ensure_cached(self, url: String, filename: String) raises -> String:
        var dest = join(self.cache_dir, filename)

        if not exists(dest):
            var urllib = Python.import_module("urllib.request")
            print("Downloading", filename, "...")
            _ = urllib.urlretrieve(url, dest)
            print("Saved to", dest)

        return dest

    def _ensure_raw_if_needed(self, url: String, raw_name: String, out_path: String) raises -> Optional[String]:
        """Returns the cached raw file path if out_path still needs to be derived, else None."""
        if exists(out_path):
            return None
        return self.ensure_cached(url, raw_name)

    def ensure_cached_idx(self, url: String, gz_name: String, out_name: String, scale: Float32 = 1.0) raises -> String:
        var out_path = join(self.cache_dir, out_name)
        var raw_path = self._ensure_raw_if_needed(url, gz_name, out_path)
        if raw_path:
            self._convert_idx(raw_path.value(), out_path, scale)
        return out_path

    def _convert_idx(self, gz_path: String, out_path: String, scale: Float32) raises:
        def gunzip(src: String, dst: String) raises:
            var gzip = Python.import_module("gzip")
            var pathlib = Python.import_module("pathlib")
            _ = pathlib.Path(dst).write_bytes(gzip.open(src, "rb").read())

        var raw_path = gz_path + ".raw"
        gunzip(gz_path, raw_path)
        var raw = Path(raw_path).read_bytes()
        remove(raw_path)

        # IDX header: 2 zero bytes, dtype byte, ndims byte, then ndims x 4-byte
        # big-endian dim sizes (8 bytes for IDX1 labels, 16 for IDX3 images, etc).
        # Output is always written flat, so the shape itself doesn't matter here.
        var header_bytes = 4 + 4 * Int(raw[3])

        var out = List[Float32](capacity=len(raw) - header_bytes)
        for i in range(header_bytes, len(raw)):
            out.append(Float32(Int(raw[i])) * scale)

        with open(out_path, "w") as f:
            var bytes = Span[Byte, origin_of(out)](
                unsafe_ptr=out.unsafe_ptr().unsafe_bitcast[Byte](), length=len(out) * 4
            )
            f.write_all(bytes)

    def ensure_cached_text_encoded(
        self,
        url: String,
        txt_name: String,
        out_name: String,
        vocab_name: String,
    ) raises -> String:
        var out_path = join(self.cache_dir, out_name)
        var vocab_path = join(self.cache_dir, vocab_name)

        # Both files are written together by _encode_text, so re-encode unless
        # both are present. A stale tokens.bin from an older cache shouldn't
        # cause vocab_path to be silently skipped.
        if exists(out_path) and exists(vocab_path):
            return out_path

        var txt_path = self.ensure_cached(url, txt_name)
        self._encode_text(txt_path, out_path, vocab_path)
        return out_path

    def _encode_text(self, txt_path: String, out_path: String, vocab_path: String) raises:
        var text: String
        with open(txt_path, "r") as f:
            text = f.read()

        var seen = Set[Int]()
        for cp in text.codepoints():
            seen.add(Int(cp))

        var vocab = List[Int]()
        for cp in seen:
            vocab.append(cp)
        sort(vocab)

        var stoi = Dict[Int, Int64]()
        for i in range(len(vocab)):
            stoi[vocab[i]] = Int64(i)

        with open(vocab_path, "w") as vf:
            for cp in vocab:
                vf.write_string(String(cp))
                vf.write_string("\n")

        var ids = List[Int64](capacity=text.byte_length())
        for cp in text.codepoints():
            ids.append(stoi[Int(cp)])

        with open(out_path, "w") as outf:
            var bytes = Span[Byte, origin_of(ids)](
                unsafe_ptr=ids.unsafe_ptr().unsafe_bitcast[Byte](), length=len(ids) * 8
            )
            outf.write_all(bytes)

    def file_len(self, path: String) raises -> Int:
        return getsize(path)
