"""
Projeto: Classificação de Textos por Espaço Vetorial
Tema 5 — Processamento de Linguagem Natural
Por: Amanda Costa Borges, Nathália Oliveira dos Santos e Marcelo Heiffig Júnior
"""

import math
import re
import unicodedata


# 1. BASE DE DOCUMENTOS DE TREINAMENTO
# Cada documento é um texto associado a uma das duas categorias.


DOCUMENTOS_TREINAMENTO = [
    {"texto": "Cientistas descobriram um novo exoplaneta que pode apresentar condicoes favoraveis para a existencia de agua", "categoria": "Descobertas científicas"},
    {"texto": "Pesquisadores anunciaram a descoberta de uma nova especie marinha durante uma expedicao no oceano", "categoria": "Descobertas científicas"},
    {"texto": "Uma equipe de astronomos identificou uma galaxia distante com caracteristicas nunca observadas antes", "categoria": "Descobertas científicas"},
    {"texto": "Estudo cientifico revelou um material mais resistente que pode contribuir para novas tecnologias", "categoria": "Descobertas científicas"},
    {"texto": "Cientistas desenvolveram uma tecnica para detectar sinais de uma doenca em estagio inicial", "categoria": "Descobertas científicas"},
    {"texto": "A atriz recebeu um premio durante a cerimonia de cinema e agradeceu aos fas pelo apoio", "categoria": "Notícias de celebridades"},
    {"texto": "O cantor anunciou o lancamento de um novo album e divulgou as datas da proxima turne", "categoria": "Notícias de celebridades"},
    {"texto": "A estrela de televisao participou de uma entrevista e comentou seus proximos projetos", "categoria": "Notícias de celebridades"},
    {"texto": "O casal famoso foi fotografado durante um evento e movimentou as redes sociais", "categoria": "Notícias de celebridades"},
    {"texto": "A celebridade confirmou sua participacao em um novo filme ao lado de outros atores conhecidos", "categoria": "Notícias de celebridades"}
]


# 2. CONFIGURAÇÕES E PRÉ-PROCESSAMENTO DOS TEXTOS
# A lista de palavras comuns evita que termos muito frequentes e pouco
# informativos tenham grande influência na classificação.


PALAVRAS_COMUNS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "entre", "e",
    "ou", "que", "se", "ao", "aos", "mais", "muito",
    "teve"
}


def remover_acentos(texto):
    """Retira acentos para padronizar palavras."""
    texto_normalizado = unicodedata.normalize("NFD", texto)
    return "".join(
        caractere for caractere in texto_normalizado
        if unicodedata.category(caractere) != "Mn"
    )


def tokenizar(texto):
    """
    Converte o texto em uma lista de palavras.
    Padroniza letras, remove acentos e ignora pontuação e palavras comuns.
    """
    texto = remover_acentos(texto.lower())
    palavras = re.findall(r"[a-z0-9]+", texto)
    return [
        palavra for palavra in palavras
        if palavra not in PALAVRAS_COMUNS
    ]


# 3. CONSTRUÇÃO DO VOCABULÁRIO


def construir_vocabulario(documentos):
    """
    Cria uma lista ordenada com todas as palavras distintas
    encontradas nos documentos de treinamento.
    """
    palavras = set()

    for documento in documentos:
        palavras.update(tokenizar(documento["texto"]))

    return sorted(palavras)



# 4. CÁLCULO DA FREQUÊNCIA DOS TERMOS (TF)


def calcular_tf(texto, vocabulario):
    """
    Calcula a frequência normalizada de cada termo.
    TF = quantidade do termo no texto / total de termos do texto.
    """
    palavras = tokenizar(texto)
    total_palavras = len(palavras)

    frequencias = []

    for termo in vocabulario:
        quantidade = palavras.count(termo)

        if total_palavras > 0:
            frequencias.append(quantidade / total_palavras)
        else:
            frequencias.append(0.0)

    return frequencias



# 5. CÁLCULO DA FREQUÊNCIA INVERSA DE DOCUMENTOS (IDF)


def calcular_idf(documentos, vocabulario):
    """
    Calcula IDF com suavização:
    IDF(t) = log((N + 1) / (DF(t) + 1)) + 1

    N: quantidade de documentos de treinamento.
    DF(t): quantidade de documentos que contêm o termo t.
    """
    quantidade_documentos = len(documentos)
    valores_idf = []

    for termo in vocabulario:
        documentos_com_termo = 0

        for documento in documentos:
            palavras = set(tokenizar(documento["texto"]))

            if termo in palavras:
                documentos_com_termo += 1

        idf = math.log(
            (quantidade_documentos + 1) /
            (documentos_com_termo + 1)
        ) + 1

        valores_idf.append(idf)

    return valores_idf



# 6. CONSTRUÇÃO DOS VETORES TF-IDF E DA MATRIZ TERMO-DOCUMENTO


def calcular_tfidf(texto, vocabulario, valores_idf):
    """
    Calcula o vetor TF-IDF de um texto:
    TF-IDF(t, d) = TF(t, d) * IDF(t).
    """
    valores_tf = calcular_tf(texto, vocabulario)
    return [
        valores_tf[indice] * valores_idf[indice]
        for indice in range(len(vocabulario))
    ]


def construir_matriz_tfidf(documentos, vocabulario, valores_idf):
    """
    Constrói a matriz termo-documento TF-IDF,
    onde cada linha representa um documento e cada coluna representa um termo.
    """
    matriz = []

    for documento in documentos:
        vetor = calcular_tfidf(
            documento["texto"],
            vocabulario,
            valores_idf
        )
        matriz.append(vetor)

    return matriz


# 7. CÁLCULO DOS VETORES CENTRÓIDES


def calcular_centroides(documentos, matriz_tfidf):
    """
    Calcula o centróide de cada categoria pela média dos vetores
    TF-IDF dos documentos pertencentes à categoria.
    """
    categorias = sorted({
        documento["categoria"] for documento in documentos
    })

    centroides = {}

    for categoria in categorias:
        vetores_categoria = []

        for indice, documento in enumerate(documentos):
            if documento["categoria"] == categoria:
                vetores_categoria.append(matriz_tfidf[indice])

        quantidade = len(vetores_categoria)
        dimensoes = len(matriz_tfidf[0])
        vetor_media = []

        for coluna in range(dimensoes):
            soma = sum(
                vetor[coluna] for vetor in vetores_categoria
            )
            vetor_media.append(soma / quantidade)

        centroides[categoria] = vetor_media

    return centroides



# 8. PRODUTO INTERNO, NORMA E SIMILARIDADE DE COSSENOS


def produto_interno(vetor_a, vetor_b):
    """Calcula o produto interno entre dois vetores."""
    return sum(
        vetor_a[indice] * vetor_b[indice]
        for indice in range(len(vetor_a))
    )


def calcular_norma(vetor):
    """Calcula a norma euclidiana de um vetor."""
    return math.sqrt(produto_interno(vetor, vetor))


def similaridade_cosseno(vetor_a, vetor_b):
    """
    Calcula cos(theta) = (A . B) / (||A|| * ||B||).
    Retorna zero caso algum vetor tenha norma zero.
    """
    norma_a = calcular_norma(vetor_a)
    norma_b = calcular_norma(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return produto_interno(vetor_a, vetor_b) / (norma_a * norma_b)



# 9. CLASSIFICAÇÃO DE UM NOVO TEXTO


def classificar_texto(texto, vocabulario, valores_idf, centroides):
    """
    Transforma o texto de entrada em vetor TF-IDF, compara-o com
    cada centróide e escolhe a categoria de maior similaridade.
    """
    vetor_teste = calcular_tfidf(
        texto,
        vocabulario,
        valores_idf
    )

    similaridades = {}

    for categoria, vetor_centroide in centroides.items():
        similaridades[categoria] = similaridade_cosseno(
            vetor_teste,
            vetor_centroide
        )

    if not tokenizar(texto):
        categoria_final = "Não foi possível classificar"
    elif max(similaridades.values(), default=0.0) == 0.0:
        categoria_final = "Não foi possível classificar"
    else:
        categoria_final = max(
            similaridades,
            key=similaridades.get
        )

    return vetor_teste, similaridades, categoria_final


# 10. FUNÇÕES PARA EXIBIÇÃO DOS RESULTADOS


def exibir_vetor(rotulo, vetor, vocabulario):
    """Exibe os valores de um vetor associados aos termos do vocabulário."""
    print(f"\n{rotulo}")

    for termo, valor in zip(vocabulario, vetor):
        if valor != 0:
            print(f"  {termo:<18} {valor:.4f}")


def exibir_matriz(documentos, matriz_tfidf, vocabulario):
    """
    Exibe a matriz termo-documento.
    Cada linha é um documento e cada coluna é um termo do vocabulário.
    """
    print("\nMATRIZ TERMO-DOCUMENTO (TF-IDF)")
    print("As colunas seguem a ordem do vocabulário.")

    cabecalho = "Documento".ljust(12)
    for termo in vocabulario:
        cabecalho += termo[:10].rjust(11)
    print(cabecalho)

    for indice, vetor in enumerate(matriz_tfidf):
        linha = f"D{indice + 1}".ljust(12)

        for valor in vetor:
            linha += f"{valor:.3f}".rjust(11)

        print(linha)


def exibir_resultado(texto, vetor, similaridades, categoria):
    """Apresenta o texto, o vetor calculado e a classificação."""
    print("\n" + "=" * 60)
    print("RESULTADO DA CLASSIFICAÇÃO")
    print("=" * 60)
    print(f"Texto analisado: {texto}")

    print("\nSimilaridade de cossenos com os centróides:")
    for nome_categoria, valor in similaridades.items():
        print(f"  {nome_categoria:<15}: {valor:.4f}")

    print(f"\nCategoria identificada: {categoria}")
    print("=" * 60)


# 11. TESTES DE EXEMPLO


def executar_testes(vocabulario, valores_idf, centroides):
    """Executa dois casos de teste e apresenta as classificações."""
    casos_teste = [
        {
            "texto": "Astronomos descobriram um planeta distante com agua e novas caracteristicas",
            "esperado": "Descobertas científicas"
        },
        {
            "texto": "A cantora anunciou um novo album e uma turne para seus fas",
            "esperado": "Notícias de celebridades"
        }
    ]

    print("\n" + "=" * 60)
    print("TESTES DE EXEMPLO")
    print("=" * 60)

    for numero, caso in enumerate(casos_teste, start=1):
        _, similaridades, resultado = classificar_texto(
            caso["texto"],
            vocabulario,
            valores_idf,
            centroides
        )

        print(f"\nTeste {numero}")
        print(f"Entrada: {caso['texto']}")
        print(f"Categoria esperada: {caso['esperado']}")
        print(f"Categoria obtida:   {resultado}")
        print(
            "Status: "
            + ("APROVADO" if resultado == caso["esperado"] else "REVISAR")
        )

        for categoria, valor in similaridades.items():
            print(f"  Similaridade com {categoria}: {valor:.4f}")


# 12. FUNÇÃO PRINCIPAL DO PROGRAMA


def main():
    """Executa o treinamento, mostra os dados e classifica textos."""
    print("=" * 60)
    print("CLASSIFICADOR DE NOTÍCIAS POR ESPAÇO VETORIAL")
    print("Descobertas científicas x Notícias de celebridades\nTema 5 — TF-IDF e Similaridade de Cossenos")
    print("=" * 60)

    # Constrói o espaço vetorial a partir dos documentos de treinamento.
    vocabulario = construir_vocabulario(DOCUMENTOS_TREINAMENTO)

    # Calcula os pesos IDF usando somente a base de treinamento.
    valores_idf = calcular_idf(
        DOCUMENTOS_TREINAMENTO,
        vocabulario
    )

    # Cria a matriz TF-IDF dos documentos de treinamento.
    matriz_tfidf = construir_matriz_tfidf(
        DOCUMENTOS_TREINAMENTO,
        vocabulario,
        valores_idf
    )

    # Calcula um centróide para cada categoria.
    centroides = calcular_centroides(
        DOCUMENTOS_TREINAMENTO,
        matriz_tfidf
    )

    # Apresenta o vocabulário e a matriz construída.
    print("\nVOCABULÁRIO:")
    print(", ".join(vocabulario))

    exibir_matriz(
        DOCUMENTOS_TREINAMENTO,
        matriz_tfidf,
        vocabulario
    )

    # Exibe os vetores centróides de cada categoria.
    print("\nVETORES CENTRÓIDES:")
    for categoria, vetor in centroides.items():
        exibir_vetor(
            f"Centróide — {categoria}",
            vetor,
            vocabulario
        )

    # Executa dois exemplos para demonstrar o comportamento esperado.
    executar_testes(vocabulario, valores_idf, centroides)

    # Permite que o usuário classifique textos livremente.
    print("\nDigite notícias para classificar.")
    print("Digite 'sair' para encerrar o programa.")

    while True:
        texto_usuario = input("\nDigite o texto da notícia: ").strip()

        if texto_usuario.lower() == "sair":
            print("\nPrograma encerrado.")
            break

        if not texto_usuario:
            print("Digite algum conteúdo antes de continuar.")
            continue

        vetor_teste, similaridades, categoria = classificar_texto(
            texto_usuario,
            vocabulario,
            valores_idf,
            centroides
        )

        exibir_vetor(
            "Vetor TF-IDF do texto de entrada:",
            vetor_teste,
            vocabulario
        )

        exibir_resultado(
            texto_usuario,
            vetor_teste,
            similaridades,
            categoria
        )