import camelot
import pandas as pd

# Caminho do PDF (ajuste para a pasta onde o arquivo está no seu PC)
pdf_path = "Estoque Porto Alegre 31102024.pdf"

# Saída final
out_xlsx = "estoque_porto_alegre_completo.xlsx"

# Extração de tabelas - flavor="stream" geralmente funciona melhor para inventários
tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

# Concatenar todas as tabelas em um único DataFrame
dfs = [t.df for t in tables]
df_concat = pd.concat(dfs, ignore_index=True)

# Renomear colunas (ajustando para seu padrão)
df_concat.columns = [
    "COD", "TIPO", "SUB DESCRICAO", "MARCA", "UN", 
    "ANTIGO", "GTIN", "NCM", "QTD.", "PRECO", "VALOR TOTAL"
]

# Remover a linha de cabeçalho repetida que pode aparecer no meio do DataFrame
df_concat = df_concat[df_concat["COD"] != "COD"]

# Converter numéricos (substitui vírgula por ponto e transforma em float/int)
def to_number(x):
    try:
        return float(str(x).replace(".", "").replace(",", "."))
    except:
        return x

df_concat["QTD."] = df_concat["QTD."].apply(to_number)
df_concat["PRECO"] = df_concat["PRECO"].apply(to_number)
df_concat["VALOR TOTAL"] = df_concat["VALOR TOTAL"].apply(to_number)

# Salvar no Excel
df_concat.to_excel(out_xlsx, index=False)

print(f"Planilha gerada com sucesso: {out_xlsx}")
