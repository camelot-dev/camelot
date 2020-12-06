import camelot


tables = camelot.read_pdf("tests/files/foo_image.pdf", flavor="lattice_ocr")
print(tables[0].df)

# camelot.plot(tables[0], kind="contour").show()
# camelot.plot(tables[0], kind="grid").show()

