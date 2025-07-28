import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def extrair_imoveis(url_base, paginas=1):
    imoveis = []

    for pagina in range(1, paginas + 1):
        url = f"{url_base}?pagina={pagina}"
        print(f"Acessando: {url}")
        resposta = requests.get(url)
        if resposta.status_code != 200:
            print(f"Erro ao acessar a página {pagina}")
            continue

        soup = BeautifulSoup(resposta.text, 'html.parser')
        anuncios = soup.find_all('article', class_='property-card__container')

        for anuncio in anuncios:
            try:
                titulo = anuncio.find('span', class_='property-card__title').get_text(strip=True)
                endereco = anuncio.find('span', class_='property-card__address').get_text(strip=True)
                preco = anuncio.find('div', class_='property-card__price').get_text(strip=True)

                detalhes = anuncio.find_all('li', class_='property-card__detail-item')
                quartos = detalhes[0].get_text(strip=True) if len(detalhes) > 0 else ''
                banheiros = detalhes[1].get_text(strip=True) if len(detalhes) > 1 else ''
                vagas = detalhes[2].get_text(strip=True) if len(detalhes) > 2 else ''

                imoveis.append({
                    'Título': titulo,
                    'Endereço': endereco,
                    'Preço': preco,
                    'Quartos': quartos,
                    'Banheiros': banheiros,
                    'Vagas': vagas
                })
            except Exception as e:
                print(f"Erro ao processar um anúncio: {e}")
                continue

        time.sleep(1)  # Respeita o tempo entre requisições

    return pd.DataFrame(imoveis)

# URL base para imóveis em Ourinhos
url_base = 'https://www.vivareal.com.br/aluguel/sp/ourinhos/'

# Número de páginas que deseja extrair
numero_paginas = 2

# Executa o scraping
df_imoveis = extrair_imoveis(url_base, paginas=numero_paginas)

# Salva os dados em um arquivo CSV
df_imoveis.to_csv('imoveis_ourinhos.csv', index=False, encoding='utf-8')
print("Dados salvos em 'imoveis_ourinhos.csv'")