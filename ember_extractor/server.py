from http.server import BaseHTTPRequestHandler, HTTPServer

import ember_features
import numpy as np

OriginalHasher = ember_features.FeatureHasher


class CompatibleHasher(OriginalHasher):
    def transform(self, samples):
        if self.input_type == "string" and len(samples) == 1 and isinstance(samples[0], str):
            samples = [[samples[0]]]
        return super().transform(samples)


ember_features.FeatureHasher = CompatibleHasher
extractor = ember_features.PEFeatureExtractor(feature_version=2, print_feature_warning=False)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/features":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(411)
            return
        if size <= 0 or size > 20 * 1024 * 1024:
            self.send_error(413)
            return
        self.connection.settimeout(10)
        data = self.rfile.read(size)
        if len(data) != size:
            self.send_error(400)
            return
        try:
            vector = extractor.feature_vector(data)
            if vector.shape != (2381,) or not np.isfinite(vector).all():
                raise ValueError("invalid feature vector")
        except Exception:
            self.send_error(422)
            return
        body = vector.astype("<f4", copy=False).tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("0.0.0.0", 8090), Handler).serve_forever()
