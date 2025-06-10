import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Lista de IDs de vídeos escolhidos
VIDEOS_ESCOLHIDOS = ['jz_As_nzSUA','U4VlAKE_fMg','J7IysernjcQ', 'TUy6SC2MRig',
                         'Xeugx_L24V0', 'eMzxhNRQV4A', 'ZPmkNvM5yMQ', 'cUNI6-3aYzg',
                         'AMmh1siH4mY', 'pbb0jzXt_xA', 'CANYM34cIuQ', '2d0waLuCjzI', 'PMs7KDyRo6Y', 'YZldtLXNfu8', '5tNRcuzc4YU',
                         'Xnp8g_DrKmM', 'cnZs8nAkdrg', 'aw9lP3WAl2o', '2fcxZTyaREk', 'posTc56basM', 'arJVNwjhmRY', 'z2oF6Itzzyo',
                         'nK_h08KTYlo', 'Qs35xCxWwVI', 'jizAto4-ofs', 'Y_befLlahys', 'gj9R3nmB67Q', 'N7UYvWskkcc', 'wW6OACofjRM',
                         'qUWs5sZE4xc', 'V3NMxTkR6n4', '0YpxVIvQ-lU', '6QOkLu4kOOI', 
                         'nTmWFWWbzkQ', '-UmOPQRpRIE', 'XSHXOEoB8jk',
                         'GgmlGTFrD3g', 'MGALB4b3O5I', 'N4AfqCBYHGo', 'xucKeCMFeWM', 'H3Mc4126A2s',
                         'IEPemH_lwLM', 'uAPy2sl-h7o', 'DA6AbROn9Fo', 'wbVPvVIepXw',
                         'y-_Ly5Tqggc', 'eSrObbUHbTQ', 'ZWYheuFOq_g', 'hO_tjm9i32g', 'UlY2deWLyw0', 'qVHPy7Np9rE'            

    ] 
# DataFrame global para armazenar os títulos para fácil acesso
df_titulos = pd.DataFrame()


def gerarTrancricoes():
    """
    Coleta as transcrições dos vídeos do YouTube e as salva em um arquivo CSV.
    """
    def obter_transcricao(video_id, idiomas=['pt', 'en']):
        for idioma in idiomas:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[idioma])
                texto = " ".join([re.sub(r'[.*?\\]', '', entry['text']) for entry in transcript])
                return " ".join(texto.split())
            except Exception:
                continue
        return None


    videos_escolhidos = ['jz_As_nzSUA','U4VlAKE_fMg','J7IysernjcQ', 'TUy6SC2MRig',
                         'Xeugx_L24V0', 'eMzxhNRQV4A', 'ZPmkNvM5yMQ', 'cUNI6-3aYzg',
                         'AMmh1siH4mY', 'pbb0jzXt_xA', 'CANYM34cIuQ', '2d0waLuCjzI', 'PMs7KDyRo6Y', 'YZldtLXNfu8', '5tNRcuzc4YU',
                         'Xnp8g_DrKmM', 'cnZs8nAkdrg', 'aw9lP3WAl2o', '2fcxZTyaREk', 'posTc56basM', 'arJVNwjhmRY', 'z2oF6Itzzyo',
                         'nK_h08KTYlo', 'Qs35xCxWwVI', 'jizAto4-ofs', 'Y_befLlahys', 'gj9R3nmB67Q', 'N7UYvWskkcc', 'wW6OACofjRM',
                         'qUWs5sZE4xc', 'V3NMxTkR6n4', '0YpxVIvQ-lU', '6QOkLu4kOOI', 
                         'nTmWFWWbzkQ', '-UmOPQRpRIE', 'XSHXOEoB8jk',
                         'GgmlGTFrD3g', 'MGALB4b3O5I', 'N4AfqCBYHGo', 'xucKeCMFeWM', 'H3Mc4126A2s',
                         'IEPemH_lwLM', 'uAPy2sl-h7o', 'DA6AbROn9Fo', 'wbVPvVIepXw',
                         'y-_Ly5Tqggc', 'eSrObbUHbTQ', 'ZWYheuFOq_g', 'hO_tjm9i32g', 'UlY2deWLyw0', 'qVHPy7Np9rE'            

    ] 

    dados = []
    ids_de_videos_nao_transcritos = []

    print("Iniciando coleta de transcrições...")
    for i, vid in enumerate(VIDEOS_ESCOLHIDOS):
        print(f"Coletando transcrição para vídeo {i+1}/{len(VIDEOS_ESCOLHIDOS)}: {vid}")
        trans = obter_transcricao(vid)
        if trans:
            dados.append({"video_id": vid, "transcricao": trans})
        else:
            ids_de_videos_nao_transcritos.append(vid)

    print(f"\nTranscricões coletadas: {len(dados)}")
    if ids_de_videos_nao_transcritos:
        print(f"IDs não transcritos (sem legendas disponíveis nos idiomas solicitados): {ids_de_videos_nao_transcritos}")

    dataframe_dados = pd.DataFrame(dados)
    dataframe_dados.to_csv("transcricoes.csv", index=False)
    print("Transcricões salvas em 'transcricoes.csv'.")


def transcricoesEmNumeroCountVectorizer():
    """
    Processa as transcrições usando a técnica Bag of Words (BoW) com CountVectorizer.
    Apenas exibe o shape da matriz resultante.
    """
    try:
        dataframe_dados = pd.read_csv('transcricoes.csv')
        textos = dataframe_dados['transcricao'].tolist()
        vetorizador = CountVectorizer(stop_words='english') # Você pode usar 'portuguese' se preferir
        matriz_de_contagem_vetorizado = vetorizador.fit_transform(textos)
        print(f"Matriz BoW: {matriz_de_contagem_vetorizado.shape} -> (n_transcricoes, n_palavras_no_vocabulario)")
    except FileNotFoundError:
        print("Erro: 'transcricoes.csv' não encontrado. Por favor, execute a opção 1 primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar BoW: {e}")


def trasncricoesEmNumeroSentenceTransformer():
    """
    Processa as transcrições usando Embeddings com SentenceTransformer.
    Apenas exibe o shape do array resultante.
    """
    try:
        dataframe = pd.read_csv('transcricoes.csv')
        textos = dataframe['transcricao'].tolist()
        print("Carregando modelo SentenceTransformer 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(textos, show_progress_bar=True)
        print(f"Shape dos embeddings: {embeddings.shape} -> (n_transcricoes, dimensão_do_embedding)")
    except FileNotFoundError:
        print("Erro: 'transcricoes.csv' não encontrado. Por favor, execute a opção 1 primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro ao gerar embeddings: {e}")


def knn_bow():
    """
    Aplica o algoritmo K-Means aos dados representados por Bag of Words (BoW).
    Salva os resultados em 'clusters_bow.csv' e exibe um gráfico PCA.
    """
    try:
        dataframe = pd.read_csv('transcricoes.csv')
        textos = dataframe['transcricao'].tolist()
        video_ids = dataframe['video_id'].tolist()

        print("\n--- Agrupamento com Bag of Words (CountVectorizer) ---")
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(textos)

        n_grupos = 10
        print(f"Aplicando K-Means com {n_grupos} clusters...")
        kmeans = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
        kmeans.fit(X)

        qualidade_divisao = kmeans.inertia_
        print(f"Inertia (BoW): {qualidade_divisao:.2f}")

        if X.shape[0] > n_grupos:
            X_dense = X.toarray()
            score = silhouette_score(X_dense, kmeans.labels_)
            print(f"Silhouette Score (BoW): {score:.2f}")

        resultado = pd.DataFrame({
            'video_id': video_ids,
            'grupo': kmeans.labels_
        })
        resultado.to_csv('clusters_bow.csv', index=False)
        print("CSV de clusters gerado: 'clusters_bow.csv'.")

        # Visualização PCA
        reduzir_matriz = PCA(n_components=2)
        X_reduced = reduzir_matriz.fit_transform(X.toarray())
        plt.figure(figsize=(10, 8))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap='tab10', s=50, alpha=0.7)
        plt.title(f"Clusters de Vídeos (K-Means) - Bag of Words")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.colorbar(label='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Erro: 'transcricoes.csv' não encontrado. Por favor, execute a opção 1 primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro no agrupamento BoW: {e}")


def knn_embe():
    """
    Aplica o algoritmo K-Means aos dados representados por Embeddings.
    Salva os resultados em 'clusters_embeddings.csv' e exibe um gráfico PCA.
    """
    try:
        dataframe = pd.read_csv('transcricoes.csv')
        textos = dataframe['transcricao'].tolist()
        video_ids = dataframe['video_id'].tolist()

        print("\n--- Agrupamento com Embeddings (SentenceTransformer) ---")
        print("Gerando embeddings para as transcrições...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        X = model.encode(textos, show_progress_bar=True)

        n_grupos = 10
        print(f"Aplicando K-Means com {n_grupos} clusters...")
        kmeans = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
        kmeans.fit(X)

        qualidade_divisao = kmeans.inertia_
        print(f"Inertia (Embeddings): {qualidade_divisao:.2f}")

        if X.shape[0] > n_grupos:
            score = silhouette_score(X, kmeans.labels_)
            print(f"Silhouette Score (Embeddings): {score:.2f}")

        resultado = pd.DataFrame({
            'video_id': video_ids,
            'grupo': kmeans.labels_
        })
        resultado.to_csv('clusters_embeddings.csv', index=False)
        print("CSV de clusters gerado: 'clusters_embeddings.csv'.")

        # Visualização PCA
        reduzir_matriz = PCA(n_components=2)
        X_reduced = reduzir_matriz.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap='tab10', s=50, alpha=0.7)
        plt.title(f"Clusters de Vídeos (K-Means) - Embeddings")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.colorbar(label='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Erro: 'transcricoes.csv' não encontrado. Por favor, execute a opção 1 primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro no agrupamento Embeddings: {e}")


def obter_titulos_videos():
    """
    Obtém os títulos dos vídeos a partir de seus IDs usando Selenium e
    os salva em 'titulos_youtube.csv', além de carregá-los em memória.
    """
    global df_titulos
    print("\nIniciando coleta de títulos de vídeos do YouTube (pode demorar um pouco)...")
    options = Options()
    options.add_argument('--headless')  # Rodar em modo invisível (sem abrir janela do navegador)
    options.add_argument('--log-level=3') # Suprimir logs do WebDriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    try:
        df_transcricoes = pd.read_csv('transcricoes.csv')
        video_ids_to_fetch = df_transcricoes['video_id'].tolist()
    except FileNotFoundError:
        print("Erro: 'transcricoes.csv' não encontrado. Por favor, execute a opção 1 (Gerar Transcrições) primeiro.")
        driver.quit()
        return

    titulos_coletados = []
    with open('titulos_youtube.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'title'])

        for i, video_id in enumerate(video_ids_to_fetch):
            url = f'https://www.youtube.com/watch?v={video_id}' # URL correta do YouTube
            try:
                driver.get(url)
                time.sleep(3)  # Espera para garantir que o título carregue
                title = driver.execute_script("return document.title")
                # Limpa o título (remove " - YouTube")
                if title.endswith(" - YouTube"):
                    title = title[:-len(" - YouTube")]
                titulos_coletados.append({'video_id': video_id, 'title': title})
                writer.writerow([video_id, title])
                print(f"[✔] {i+1}/{len(video_ids_to_fetch)}: {video_id} -> {title}")
            except Exception as e:
                print(f"[✗] Erro ao obter título para {video_id}: {e}")
                titulos_coletados.append({'video_id': video_id, 'title': 'N/A'}) # Marca como não disponível
                writer.writerow([video_id, 'N/A'])
    driver.quit()
    print("Coleta de títulos concluída. Salvo em 'titulos_youtube.csv'.")

    df_titulos = pd.DataFrame(titulos_coletados)


def analisar_grupos(metodo):
    """
    Carrega os resultados do agrupamento e exibe os títulos dos vídeos por grupo,
    permitindo ao usuário rotular cada grupo.
    """
    try:
        if metodo == 'bow':
            df_clusters_path = 'clusters_bow.csv'
            print("\n--- Análise dos Grupos (Bag of Words) ---")
        elif metodo == 'embeddings':
            df_clusters_path = 'clusters_embeddings.csv'
            print("\n--- Análise dos Grupos (Embeddings) ---")
        else:
            print("Método inválido. Escolha 'bow' ou 'embeddings'.")
            return

        df_clusters = pd.read_csv(df_clusters_path)

        if df_titulos.empty:
            print("Os títulos dos vídeos não foram carregados. Por favor, execute a opção 2 (Obter Títulos dos Vídeos) primeiro.")
            return

        # Mescla os dados de clusters com os títulos dos vídeos
        df_merged = pd.merge(df_clusters, df_titulos, on='video_id', how='left')

        for grupo_id in sorted(df_merged['grupo'].unique()):
            print(f"\n--- Grupo {grupo_id} ---")
            videos_no_grupo = df_merged[df_merged['grupo'] == grupo_id]
            if not videos_no_grupo.empty:
                for _, row in videos_no_grupo.iterrows():
                    print(f"  - ID: {row['video_id']}, Título: {row['title']}")
            else:
                print("  Nenhum vídeo neste grupo.")

            label = input(f"Sugerir um rótulo para o Grupo {grupo_id} (ex: Tecnologia, Esportes, etc. Pressione Enter para pular): ")
            if label:
                print(f"  Rótulo Sugerido para o Grupo {grupo_id}: {label}")
                # Se quiser salvar os rótulos sugeridos no futuro, armazene-os em um dicionário aqui.

        print("\n--- Análise de Grupos Concluída ---")

        # Salva os dados mesclados (video_id, title, grupo) no CSV
        resultado = df_merged[['video_id', 'title', 'grupo']]
        resultado.to_csv('titulo_grupos.csv', index=False)
        print("CSV de clusters gerado: 'titulo_grupos.csv'.")

    except FileNotFoundError:
        print(f"Erro: O arquivo de clusters para '{metodo}' não foi encontrado. Por favor, execute a opção de agrupamento (5 ou 6) primeiro.")
    except Exception as e:
        print(f"Ocorreu um erro na análise dos grupos: {e}")


def menu():
    """
    Exibe o menu principal e gerencia a execução das funções do projeto.
    """
    print("\n--- PROJETO PRÁTICO DE APRENDIZADO DE MÁQUINA ---")
    print("PLN – PROCESSAMENTO DE LINGUAGEM NATURAL - APRENDIZADO NÃO-SUPERVISIONADO - AGRUPAMENTO")
    print("\nIntegrantes: Lavínia e Michele")

    opcao = -1
    while opcao != 0:
        print("\n" + "="*30)
        print("          MENU PRINCIPAL")
        print("="*30)
        print("1 - Coletar Transcrições de Vídeos (cria 'transcricoes.csv')")
        print("2 - Obter Títulos de Vídeos (cria 'titulos_youtube.csv')")
        print("3 - (Info) Ver Shape da Matriz BoW")
        print("4 - (Info) Ver Shape dos Embeddings")
        print("5 - Aplicar K-Means e Agrupar com BoW (gera 'clusters_bow.csv' e gráfico PCA)")
        print("6 - Aplicar K-Means e Agrupar com Embeddings (gera 'clusters_embeddings.csv' e gráfico PCA)")
        print("7 - Analisar Grupos (BoW) e ver títulos")
        print("8 - Analisar Grupos (Embeddings) e ver títulos")
        print("0 - Sair")
        print("="*30)

        try:
            opcao = int(input("Digite sua opção: "))

            if opcao == 1:
                gerarTrancricoes()
            elif opcao == 2:
                obter_titulos_videos()
            elif opcao == 3:
                transcricoesEmNumeroCountVectorizer()
            elif opcao == 4:
                trasncricoesEmNumeroSentenceTransformer()
            elif opcao == 5:
                knn_bow()
            elif opcao == 6:
                knn_embe()
            elif opcao == 7:
                analisar_grupos('bow')
            elif opcao == 8:
                analisar_grupos('embeddings')
            elif opcao == 0:
                print("Saindo do programa. Até mais!")
                break
            else:
                print("Opção inválida. Por favor, digite um número válido do menu.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")

# Inicia o menu
menu()