# -*- coding: utf-8 -*-

import logging

from .poppler_backend import PopplerBackend
from .ghostscript_backend import GhostscriptBackend

logger = logging.getLogger("camelot")
backends = {"poppler": PopplerBackend, "ghostscript": GhostscriptBackend}


class ImageConversionBackend(object):
    def __init__(self, backend="poppler", use_fallback=True):
        if backend not in backends.keys():
            raise ValueError(f"Image conversion backend '{backend}' not supported")

        self.backend = backend
        self.use_fallback = use_fallback
        self.fallbacks = list(filter(lambda x: x != backend, backends.keys()))

    def convert(self, pdf_path, png_path):
        try:
            converter = backends[self.backend]()
            converter.convert(pdf_path, png_path)
        except Exception as e:
            logger.info(f"Image conversion backend '{self.backend}' failed with {str(e)}")

            if self.use_fallback:
                for fallback in self.fallbacks:
                    logger.info(f"Falling back on '{fallback}'")

                    try:
                        converter = backends[self.backend]()
                        converter.convert(pdf_path, png_path)
                    except Exception as e:
                        logger.info(f"Image conversion backend '{fallback}' failed with {str(e)}")

                        continue
                    else:
                        logger.info(f"Image conversion backend '{fallback}' succeeded")
                        break
