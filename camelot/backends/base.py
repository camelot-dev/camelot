"""Classes and functions for the ImageConversionBackend backends."""


class ConversionBackend:  # noqa D101
    def installed(self) -> bool:  # noqa D102
        raise NotImplementedError

    def convert(  # noqa D102
        self, pdf_path: str, png_path: str, resolution: int = 300
    ) -> None:  # noqa D102
        raise NotImplementedError
