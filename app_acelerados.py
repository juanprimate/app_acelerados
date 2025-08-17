import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.optimize import minimize
from scipy.stats import norm, t
from statsmodels.tools.numdiff import approx_hess
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.cm as cm


# Funções auxiliares
def validacao_dados(dados, t, aceleracao, censura):
    variables = [t, aceleracao, censura]
    missing_variables = [var for var in variables if var not in dados.columns]
    if missing_variables:
        message = f"O seu banco de dados não contém todas as variáveis necessárias: {', '.join(missing_variables)}"
        raise ValueError(message)

    if dados.isnull().any().any():
        message = "Atenção: Banco de dados possui valores ausentes, linhas com NA foram removidas."
        raise ValueError(message)

    if (dados[t] <= 0).any():
        message = "Variável tempo de falha possui valores não-positivos, por favor verifique."
        raise ValueError(message)

    if not pd.api.types.is_numeric_dtype(dados[aceleracao]):
        message = "A variável nível de aceleração contém valores não numéricos. Por favor, verifique."
        raise ValueError(message)

    unique_values = dados[censura].unique()
    if len(unique_values) != 2 or not all(value in [0, 1] for value in unique_values):
        message = "A variável de censura não contém apenas valores 0 ou 1, favor adequar."
        raise ValueError(message)

    st.write("Banco de dados validado.")
    return dados

def carregar_dados(uploaded_file, separator):
    if uploaded_file is not None:
        extension = uploaded_file.name.split('.')[-1].lower()
        if extension == "csv":
            dados = pd.read_csv(uploaded_file, sep=separator)
        elif extension == "txt":
            dados = pd.read_table(uploaded_file, sep=separator, header=True)
        else:
            return None
        return dados
    return None

#-----------------------------------------------------------------------------------------------------
# Configuração do layout do Streamlit
st.set_page_config(layout="centered")

# Definir o título e o cabeçalho
st.title("Modelo Tempo de Falha Acelerado")
st.sidebar.title("Índice")
menu = st.sidebar.radio(
    "Menu",
    ["Apresentação", "Banco de dados", "Modelagem"]
)

# Conteúdo da aba Apresentação
if menu == "Apresentação":
    st.header("Introdução")
    st.markdown("""
    <p style='text-align: justify;'>
    A finalidade dos testes acelerados é adquirir informações 
    sobre a confiabilidade de um sistema de forma rápida. Para isso, aumenta-se o nível de estresse 
    aplicado aos componentes com o auxílio de uma ou mais covariáveis (por exemplo, temperatura, 
    tensão, umidade, pressão, etc.), isso acelera a ocorrência de falhas e reduz o tempo de 
    experimentação, sem alterar a estrutura dos modos de falha de interesse. Desta forma, 
    é possível obter rapidamente dados que, devidamente modelados e analisados, fornecerão 
    informações valiosas sobre a vida útil ou o desempenho do sistema em condições de uso normais, 
    resultando em economia de tempo e recursos financeiros.
    </p>
    <p style='text-align: justify;'> A implementação destes testes acelerados oferece diversos benefícios, 
    incluindo a avaliação da confiabilidade do sistema em condições extremas, certificação de componentes cruciais, comparação
    de confiabilidade entre diferentes fabricantes para auxiliar na escolha de fornecedores, e aceleração
    do desenvolvimento do sistema, proporcionando informações rápidas para correções e reduzindo o tempo
    de lançamento no mercado. Todos esses propósitos convergem para a finalidade de obter informações 
    rápidas e confiáveis sobre a confiabilidade de sistemas e auxiliam os fabricantes a tomarem decisões
    informadas sobre o seu desenvolvimento, certificação e melhoria do mesmo.
    </p>
    """, unsafe_allow_html=True)
    
    st.header("Modelo Tempo de Falha Acelerado") # AJUSTAR ISSO
    st.latex(r'''
    \begin{align*}
    &\text{Considerando que a influência das covariáveis de aceleração atua de forma multiplicativa em }\\
    &\text{relação ao tempo, na qual é representada por um vetor } \mathbf{x} \text{ de dimensão } k \times 1. \\
    &\text{Nesse contexto, a ocorrência dos eventos de interesse seguem o modelo de tempo de falha acelerado,}\\
    &\text{com função densidade de probabilidade dada por:} \\
    &f(t | \mathbf{x}) = \frac{\kappa}{\alpha} \left( \frac{t e^{\beta_0+\beta_1\mathbf{x}} }{\alpha}\right)^{\kappa-1} e^{\beta_0+\beta_1\mathbf{x}} \exp \left( - \left( \frac{ t e^{\beta_0+\beta_1\mathbf{x}} }{\alpha}\right)^{\kappa} \right) \\
    &\text{em que } g(x) = e^{\beta_0+\beta_1\mathbf{x}} \text{ é o fator de aceleração exponencial, } h_0(t) = \frac{\kappa}{\gamma} \left( \frac{t}{\gamma}\right)^{\kappa - 1} \text{ é a função de taxa de }\\
    &\text{falha de linha de base e } H_0(t) = \left( \frac{t}{\gamma}\right)^{\kappa} \text{ é a função de taxa de falha acumulada de linha de base,}\\
    &\text{ambas com distribuição Weibull. A função de taxa de falha é descrita por:} \\
    &h(t | \mathbf{x}) = \frac{\kappa}{\alpha} \left( \frac{t e^{\beta_0+\beta_1\mathbf{x}} }{\alpha}\right)^{\kappa-1} e^{\beta_0+\beta_1\mathbf{x}}, \\
    &\text{com função taxa de falha acumulada dada por:} \\
    &H(t | \mathbf{x}) = \left( \frac{ t e^{\beta_0+\beta_1\mathbf{x}} }{\alpha}\right)^{\kappa}, \\
    &\text{e com função de confiabilidade expressa por:} \\
    &R(t | \mathbf{x}) = \exp \left( - \left( \frac{ t e^{\beta_0+\beta_1\mathbf{x}} }{\alpha}\right)^{\kappa} \right). \\
    &\text{A função de verossimilhança para o modelo tempo de falha acelerado é dada por} \\
    &L(\theta; \mathbf{t}) = \prod_{i=1}^n \left[ \frac{\kappa}{\alpha} \left( \frac{t_i e^{\beta_0+\beta_1\mathbf{x}_i} }{\alpha}\right)^{\kappa-1} e^{\beta_0+\beta_1\mathbf{x}_i} \right]^ {\delta_i} \exp \left( - \left( \frac{ t_i e^{\beta_0+\beta_1\mathbf{x}_i} }{\alpha}\right)^{\kappa} \right).
    \end{align*}
    ''')
        
    st.header("Interpretação")
    st.markdown("""
    <p style='text-align: justify;'> <b>Função de densidade de probabilidade:</b> Descreve a probabilidade de falha por unidade de tempo em um determinado tempo <b>t</b>.
    Assim, o pico da Curva indica o tempo mais provável de falha. E a forma da curva pode indicar o comportamento do sistema, isto é, uma forma simétrica em torno de um pico indica que as falhas são mais concentradas em torno de um tempo específico, enquanto uma curva mais achatada indica uma maior dispersão dos tempos de falha.</p>

    <p style='text-align: justify;'> <b>Função de Risco:</b> Descreve a taxa instantânea de falha em um determinado tempo <b>t</b>, dado que o item ainda não falhou até esse tempo.
    Essa curva pode ser crescente (indicando que o sistema se torna mais propenso a falhar à medida que o tempo passa), ou decrescente (indicando que o sistema se torna menos propenso a falhar com o passar do tempo) ou Constante (Indica que o risco de falha é independente do tempo).</p>

    <p style='text-align: justify;'> <b>Função de Risco Acumulado:</b> Descreve a probabilidade acumulada de falha até o tempo <b>t</b>.
    Uma curva suavemente crescente indica um aumento contínuo na probabilidade de falha, enquanto a inclinação da curva revela a taxa de falha ao longo do tempo; uma inclinação mais acentuada indica um aumento rápido no risco de falha.</p>

    <p style='text-align: justify;'> <b>Função de Confiabilidade:</b> Descreve a probabilidade de que um item funcione sem falhas até o tempo <b>t</b>.
    Naturalmente decrescente, a curva <b>R(t)</b> indica que a probabilidade de sobrevivência diminui com o tempo. 
    A rapidez do declínio revela a durabilidade do sistema: uma curva que decresce lentamente sugere alta confiabilidade, 
    enquanto uma curva que decresce rapidamente sugere baixa confiabilidade. </p>

    <p style='text-align: justify;'> <b>Parâmetros:</b> </p>

    <p style='text-align: justify;'> - <b>κ</b> (parâmetro de forma) indica a forma da distribuição Weibull. Se 
    <b>κ < 1</b> a taxa de falha diminui ao longo do tempo; se <b>κ = 1</b> a taxa de falha é constante; 
    se <b>κ > 1</b>, a taxa de falha aumenta ao longo do tempo.</p>

    <p style='text-align: justify;'> - <b>γ</b>  (parâmetro de escala) afeta a escala dos tempos de falha.</p>

    <p style='text-align: justify;'> - <b>β0</b> (intercepto) afeta a posição da curva no eixo do tempo.</p>

    <p style='text-align: justify;'> - <b>β1</b> (coeficientes de regressão) representam a influência das covariáveis de aceleração no tempo de falha. </p>
    """, unsafe_allow_html=True)


# ---------------------- Banco de Dados -----------------------------------------
elif menu == "Banco de dados":
    st.header("Banco de Dados")
    st.markdown("""
    <p style='text-align: justify;'> Selecione um banco de dados e faça upload de um arquivo em formato .csv para análise. </p>
    """, unsafe_allow_html=True)
    
    # Carregar arquivo CSV ou TXT
    uploaded_file = st.file_uploader("Escolha o arquivo para carregar os dados", type=["csv", "txt"])
    separator = st.selectbox("Selecione o separador:", ["Ponto e Vírgula (;)", "Vírgula (,)", "Tab (\t)"])
    separator_map = {"Ponto e Vírgula (;)": ";", "Vírgula (,)": ",", "Tab (\t)": "\t"}
    
    dados = carregar_dados(uploaded_file, separator_map[separator])

    if dados is not None:
        # Seleção de variáveis
        st.subheader("Seleção de Variáveis")
        col_tempo = st.selectbox("Tempo:", ["Selecione"] + list(dados.columns))
        col_aceleracao = st.selectbox("Aceleração:", ["Selecione"] + list(dados.columns))
        col_censura = st.selectbox("Censura:", ["Selecione"] + list(dados.columns))
        
        if col_tempo != "Selecione" and col_aceleracao != "Selecione" and col_censura != "Selecione":
            try:
                dados_ordenados = dados.copy()
                dados_ordenados["sistema"] = range(1, len(dados_ordenados) + 1)
                dados_ordenados = dados_ordenados.sort_values(by=col_tempo)
                dadostratados = validacao_dados(dados_ordenados, t=col_tempo, aceleracao=col_aceleracao, censura=col_censura)
                st.session_state['dadostratados'] = dadostratados

                st.success("Banco de dados validado.")
                st.subheader("Dados Validados")
                st.write(dadostratados)

            # ----------------- Gráfico de contagem dos dados ------------------------------------------
                st.subheader('Representação dos Dados')
                #dadostratados['acel-censura'] = dadostratados['aceleracao'].astype(str) + '_' + dadostratados['censura'].astype(str)
                # gráfico usando Seaborn (que é mais semelhante ao ggplot2)
                @st.cache_data
                def plot_representacao_dados():
                    # criar uma nova coluna combinando aceleracao e censura
                    dadostratados['acel_censura'] = dadostratados['aceleracao'].astype(str) + '_' + dadostratados['censura'].astype(str)
                    
                    plt.figure(figsize=(12,8))
                    sns.histplot(data=dadostratados, x='t', hue='acel_censura', multiple='stack', palette='pastel', bins=25, kde=False, element='bars')
                    plt.xlabel('Tempo', fontsize=14)
                    plt.ylabel('Contagem', fontsize=14)
                    plt.title('Distribuição de Tempo por Aceleração e Censura', fontsize=14)
                    
                    plt.tick_params(axis='both', which='major', labelsize=14)

                    # obtendo as combinações únicas de aceleracao e censura
                    unique_combinations = dadostratados['acel_censura'].unique()
                    # Configuração automática da legenda
                    legend_labels = unique_combinations.tolist()
                    #legend_colors = sns.color_palette('viridis', n_colors=len(legend_labels)).as_hex()
                    
                    plt.legend(title='Aceleração e Censura', labels=legend_labels, loc='upper center', fontsize=14, title_fontsize=14, facecolor='white', edgecolor='black')
                    
                    return plt.gcf()

                # chamando o gráfico - ESTE GRÁFICO PRECISA SER DISCUTIDO
                fig = plot_representacao_dados()
                st.pyplot(fig)

            # ----------------- TTTplot ----------------------------------------
                st.subheader('Gráfico TTTplot')

                def calcular_ttt_statistic_normalized(dados):
                    kmf = KaplanMeierFitter()
                    kmf.fit(dados['t'], event_observed=dados['censura'])

                    times = kmf.event_table.index.values
                    observed = kmf.event_table['observed'].values
                    at_risk = kmf.event_table['at_risk'].values

                    n = len(observed)
                    phi = np.zeros(n)
                    
                    for i in range(1, n):
                        sum_product = 0
                        for j in range(1, i+1):
                            product = 1
                            for k in range(1, j+1):
                                if at_risk[k-1] > 0:  # Evitar divisão por zero
                                    product *= (1 - observed[k-1] / at_risk[k-1])
                            if j < len(times):
                                delta_t = times[j] - times[j-1]
                            else:
                                delta_t = times[j-1] - times[j-2]
                            sum_product += product * delta_t
                        phi[i] = sum_product
                    
                    # Normalizar phi pelo valor no máximo de n
                    phi_normalized = phi / phi[-1]
                    
                    return phi_normalized

                # Carregando os dados
                dados = dadostratados
                fig, ax = plt.subplots()
                for aceleracao in dadostratados['aceleracao'].unique():
                    dados = dadostratados[dadostratados['aceleracao'] == aceleracao]
                    phi_normalized = calcular_ttt_statistic_normalized(dados)
                    
                    # Normalizar por n
                    n = len(phi_normalized)
                    i_n = np.arange(1, n+1) / n
                    
                    # Preparar os dados para o formato de escada
                    i_n = np.insert(i_n, 0, 0)
                    phi_normalized = np.insert(phi_normalized, 0, 0)
                    
                    # Plotar no formato de escada
                    ax.step(i_n, phi_normalized, where='post', label=f'Aceleração {aceleracao}')
                    ax.scatter(i_n, phi_normalized)

                ax.set_xlabel('i/n')
                ax.set_ylabel(r'$\phi_{n}$')
                ax.set_title('Estatística TTT para diferentes acelerações')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)


            except ValueError as e:
                st.error(f"Erro de validação: {e}")
    else:
        st.warning("Por favor, carregue um arquivo .csv ou .txt.")


# ---------------------------- MODELAGEM ----------------------------------------
# Conteúdo da aba Modelagem
elif menu == "Modelagem":
    st.header("Modelagem")
    if 'dadostratados' in st.session_state:
        dadostratados = st.session_state['dadostratados']
        # Função de log-verossimilhança
        def logLik(params, dados):
            gamma, kappa, b0, b1 = params
            
            delta = dados['censura']
            t = dados['t']
            x = dados['aceleracao']
            fa = np.exp(b0 + b1 * x)
            h_Aux = (kappa / gamma) * (t * fa / gamma) ** (kappa - 1) * fa
            H_Aux = (t * fa / gamma) ** kappa
            
            log_hAux = delta * np.log(h_Aux)
            log_RAux = -H_Aux
            
            ll = np.sum(log_hAux) + np.sum(log_RAux)
            if not np.isfinite(ll):
                ll = -1e5
            
            return -ll

        # Função para calcular estatísticas dos parâmetros
        def calcular_estatisticas(dados_tratados):
            def objective_function(params):
                return logLik(params, dados_tratados)

            bounds = [(0.01, None), (0.01, None), (-np.inf, None), (-np.inf, None)]
            initial_params = [1, 1, 1, 1]

            result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
            est = np.round(result.x, 4)

            #hessian_inv = result.hess_inv.todense()
            hessian = approx_hess(est, objective_function) #usando esta função, fica mais próximo do que a Edilenia obteve,
                                                            #porém não calcula erro, ic e p-valor pra beta0 e gamma (dá inválido)
            #fisher_info = np.array(hessian_inv)
            fisher_info = np.array(hessian)
            matrix_var_cov = np.linalg.inv(fisher_info)
            erro_padrao = np.round(np.sqrt(np.diag(matrix_var_cov)), 4)
            estatitica_t = est / erro_padrao

            n = len(dados_tratados['t'])
            gl = n - 1

            p_valor = 2 * t.sf(np.abs(estatitica_t), df=gl)
            IC_inf = est - norm.ppf(0.975) * erro_padrao
            IC_sup = est + norm.ppf(0.975) * erro_padrao

            resumo = pd.DataFrame({
                "Parâmetro": ["γ", "κ", "β₀", "β₁"],
                "Estimativa": est,
                "Erro Padrão": erro_padrao,
                "IC inf": IC_inf,
                "IC sup": IC_sup,
                "p-valor": p_valor
            })

            return resumo
        resumo = calcular_estatisticas(dadostratados)
        st.session_state['resumo'] = resumo
        # Tabela dos parâmetros
        st.write("Medidas Resumo")
        st.dataframe(resumo.style.format({
                    "Estimativa": "{:.4f}",
                    "Erro Padrão": "{:.4f}",
                    "IC inf": "{:.4f}",
                    "IC sup": "{:.4f}",
                    "p-valor": "{:.2e}"  # Notação científica para p-valores
                }))
        # ------------------ Plot confiabilidade ----------------------
        # Funções AFT
        def H_AFT(t, gamma, kappa, fa):
            return (t * fa / gamma) ** kappa

        def h_AFT(t, gamma, kappa, fa):
            return (kappa / gamma) * (t * fa / gamma) ** (kappa - 1) * fa
        # Função para calcular a curva de Kaplan-Meier
        def calcular_kaplan_meier(dados_tratados, col_tempo, col_censura, col_aceleracao):
            kmf = KaplanMeierFitter()

            dados_aux_list = []

            for nivel in dados_tratados[col_aceleracao].unique():
                dados_nivel = dados_tratados[dados_tratados[col_aceleracao] == nivel]
                kmf.fit(dados_nivel[col_tempo], event_observed=dados_nivel[col_censura], label=str(nivel))

                df_kmf = kmf.survival_function_
                df_kmf['time'] = df_kmf.index
                df_kmf = df_kmf.reset_index(drop=True)
                df_kmf['estimate'] = df_kmf[kmf._label]

                conf_interval = kmf.confidence_interval_
                df_kmf['conf.low'] = conf_interval.iloc[:, 0]
                df_kmf['conf.high'] = conf_interval.iloc[:, 1]
                df_kmf['n.event'] = kmf.event_table['observed'].values
                df_kmf['std.error'] = (conf_interval.iloc[:, 1] - conf_interval.iloc[:, 0]) / 3.92

                df_kmf['varS'] = df_kmf['std.error'] ** 2
                df_kmf['varH'] = df_kmf['varS'] / (df_kmf['estimate'] ** 2)
                df_kmf['H'] = -np.log(df_kmf['estimate'])
                df_kmf['liH'] = df_kmf['H'] - 1.96 * np.sqrt(df_kmf['varH'])
                df_kmf['lsH'] = df_kmf['H'] + 1.96 * np.sqrt(df_kmf['varH'])
                df_kmf['h'] = np.append([np.nan], np.diff(df_kmf['H']) / np.diff(df_kmf['time']))
                df_kmf['Ft'] = 1 - df_kmf['estimate']
                df_kmf['f'] = df_kmf['h'] * df_kmf['estimate']
                df_kmf['aceleracao'] = nivel

                dados_aux_list.append(df_kmf)

            dados_aux = pd.concat(dados_aux_list).reset_index(drop=True)
            
            return dados_aux

        
        # Função para calcular os dados estimados
        def calcular_dados_estimados(dados_tratados, resumo):
            tempos = np.concatenate([np.linspace(0, max(dados_tratados['t']), 300)] * len(dados_tratados['aceleracao'].unique()))
            aceleracao = np.repeat(dados_tratados['aceleracao'].unique(), 300)
            
            H_hat = H_AFT(tempos, resumo.iloc[0, 1], resumo.iloc[1, 1], np.exp(resumo.iloc[2, 1] + resumo.iloc[3, 1] * aceleracao))
            h_hat = h_AFT(tempos, resumo.iloc[0, 1], resumo.iloc[1, 1], np.exp(resumo.iloc[2, 1] + resumo.iloc[3, 1] * aceleracao))
            
            dados_estimados = pd.DataFrame({
                'tempos': tempos,
                'aceleracao': aceleracao,
                'H_hat': H_hat,
                'h_hat': h_hat
            })
            dados_estimados['R_hat'] = np.exp(-dados_estimados['H_hat'])
            dados_estimados['f_hat'] = dados_estimados['h_hat'] * dados_estimados['R_hat']
            
            # Calcular os intervalos de confiança para R_hat
            z = norm.ppf(0.975)
            dados_estimados['IC_inf'] = dados_estimados['R_hat'] * np.exp(-z * (dados_estimados['h_hat'] / np.sqrt(len(dados_estimados))))
            dados_estimados['IC_sup'] = dados_estimados['R_hat'] * np.exp(z * (dados_estimados['h_hat'] / np.sqrt(len(dados_estimados))))
    
            return dados_estimados
        
        # Função para plotar a confiabilidade com região de estimação
        def plot_confiabilidade(dados_aux, dados_estimados):
            fig, ax = plt.subplots()

            unique_aceleracoes = dados_aux['aceleracao'].unique()
            colors = sns.color_palette('husl', len(unique_aceleracoes))
            color_map = dict(zip(unique_aceleracoes, colors))
            
            # Plotar a curva de Kaplan-Meier e a região de confiança
            for label, df_group in dados_aux.groupby('aceleracao'):
                time = df_group['time'].values
                estimate = df_group['estimate'].values
                conf_low = df_group['conf.low'].values
                conf_high = df_group['conf.high'].values
                
                ax.step(time, estimate, where='post', label=f'Aceleração {label}', color=color_map[label], alpha=0.5)
                ax.fill_between(time, conf_low, conf_high, step='post', alpha=0.2, color=color_map[label])
                ax.scatter(df_group[df_group['n.event'] == 0]['time'], df_group[df_group['n.event'] == 0]['estimate'], marker='x', color=color_map[label])
            
            for label, df_group in dados_estimados.groupby('aceleracao'):
                ax.plot(df_group['tempos'], df_group['R_hat'], label=f'Estimado {label}', linewidth=1.4, color=color_map[label])
            
            ax.set_xlabel('Tempo')
            ax.set_ylabel('R(t)')
            ax.legend(title="Nível de aceleração")
            ax.grid(True)
            st.pyplot(fig)
        
        # Função para calcular e plotar a densidade
        def plot_densidade(dados_aux, dados_estimados):
            fig, ax = plt.subplots()

            unique_aceleracoes = dados_aux['aceleracao'].unique()
            colors = sns.color_palette('husl', len(unique_aceleracoes))
            color_map = dict(zip(unique_aceleracoes, colors))

            # Plotar a curva de densidade estimada
            for label, df_group in dados_aux.groupby('aceleracao'):
                ax.step(df_group['time'], df_group['f'], where='post', label=f'Aceleração {label}', color=color_map[label], alpha=0.5)
            
            for label, df_group in dados_estimados.groupby('aceleracao'):
                ax.plot(df_group['tempos'], df_group['f_hat'], label=f'Estimado {label}', linewidth=1.4, color=color_map[label])

            ax.set_xlabel('Tempos')
            ax.set_ylabel(r'$f(t)$')
            ax.legend(title="Nível de aceleração", loc='best')
            ax.grid(True)
            st.pyplot(fig)

        # Função para calcular o risco acumulado
        def plot_risco_acumulado(dados_aux, dados_estimados):
            fig, ax = plt.subplots()

            unique_aceleracoes = dados_aux['aceleracao'].unique()
            colors = sns.color_palette('husl', len(unique_aceleracoes))
            color_map = dict(zip(unique_aceleracoes, colors))

            # Plotar a curva de risco acumulado estimado
            for label, df_group in dados_aux.groupby('aceleracao'):
                ax.step(df_group['time'], df_group['H'], where='post', label=f'Aceleração {label}', color=color_map[label], alpha=0.5)

            for label, df_group in dados_estimados.groupby('aceleracao'):
                ax.plot(df_group['tempos'], df_group['H_hat'], label=f'Estimado {label}', linewidth=1.4, color=color_map[label])

            ax.set_xlabel('Tempo')
            ax.set_ylabel(r'$H(t)$')
            ax.legend(title="Nível de aceleração", loc='best')
            ax.grid(True)
            st.pyplot(fig)

        # Função para calcular o risco instantâneo
        def plot_risco(dados_aux, dados_estimados):
            fig, ax = plt.subplots()

            unique_aceleracoes = dados_aux['aceleracao'].unique()
            colors = sns.color_palette('husl', len(unique_aceleracoes))
            color_map = dict(zip(unique_aceleracoes, colors))

            # Plotar a curva de risco estimado
            for label, df_group in dados_aux.groupby('aceleracao'):
                ax.step(df_group['time'], df_group['h'], where='post', label=f'Aceleração {label}', color=color_map[label], alpha=0.5)

            for label, df_group in dados_estimados.groupby('aceleracao'):
                ax.plot(df_group['tempos'], df_group['h_hat'], label=f'Estimado {label}', linewidth=1.4, color=color_map[label])

            ax.set_xlabel('Tempos')
            ax.set_ylabel(r'$h(t)$')
            ax.legend(title="Nível de aceleração", loc='best')
            ax.grid(True)
            ax.set_ylim(0, 0.05)  # Define o limite do eixo y
            st.pyplot(fig)

        if 'resumo' in st.session_state:
            dados_aux = calcular_kaplan_meier(st.session_state['dadostratados'], 't', 'censura', 'aceleracao')
            st.session_state['dados_aux'] = dados_aux
            
            dados_estimados = calcular_dados_estimados(st.session_state['dadostratados'], st.session_state['resumo'])
            st.session_state['dados_estimados'] = dados_estimados


            st.header("Gráficos de ajuste")
# ----------------- Confiabilidade -------------------------
            st.subheader("Confiabilidade")
            plot_confiabilidade(st.session_state['dados_aux'], st.session_state['dados_estimados'])
# ----------------- Densidade -------------------------        
            st.subheader("Densidade")
            plot_densidade(st.session_state['dados_aux'], st.session_state['dados_estimados'])
# ----------------- Risco acumulado -------------------------
            st.subheader("Risco Acumulado")
            plot_risco_acumulado(st.session_state['dados_aux'], st.session_state['dados_estimados'])
# ----------------- Risco instantâneo -------------------------
            st.subheader("Risco Instantâneo")
            plot_risco(st.session_state['dados_aux'], st.session_state['dados_estimados'])

# ---------------- Dados residuais ---------------------------------------
        # Função para calcular os dados residuais
        def calcular_dados_residuais(dados_tratados, resumo):
            fa = np.exp(resumo.loc[2, 'Estimativa'] + resumo.loc[3, 'Estimativa'] * dados_tratados['aceleracao'])
            H_hat = (dados_tratados['t'] * fa / resumo.loc[0, 'Estimativa']) ** resumo.loc[1, 'Estimativa']

            dados_residual = dados_tratados.copy()
            dados_residual['fa'] = fa
            dados_residual['H_hat'] = H_hat
            dados_residual['R_hat'] = np.exp(-H_hat)
            dados_residual['res_quantilico'] = np.sort(norm.ppf(1 - dados_residual['R_hat']))
            dados_residual['res_cox_snell'] = H_hat
            dados_residual['res_martingal'] = dados_residual['censura'] - H_hat
            dados_residual['res_deviance'] = np.sign(dados_residual['res_martingal']) * np.sqrt(-2 * (dados_residual['res_martingal'] + dados_residual['censura'] * np.log(dados_residual['censura'] - dados_residual['res_martingal'])))
            dados_residual['quantilTeorico'] = norm.ppf(np.linspace(0.5 / len(dados_residual), 1 - 0.5 / len(dados_residual), len(dados_residual)))

            return dados_residual
        
        # Função para calcular os resíduos quantílicos
        def plot_residuos_quantilicos(dados_residual):
            # Quantil aleatorizado
            res = pd.DataFrame({
                'QA': dados_residual['res_quantilico']
            }).reset_index().rename(columns={'index': 'ID'})
            
            # Utilizando statsmodels qqplot
            qq_plot = sm.ProbPlot(dados_residual['res_quantilico'], dist=norm)
            qq_fig = qq_plot.qqplot(line='45', alpha=0.5, lw=1, marker='o')
            qq_df = pd.DataFrame({'x': qq_plot.theoretical_quantiles, 'y': qq_plot.sample_quantiles})
            qq_df['yy'] = qq_df['y']
            qq_df['y'] = qq_df['y'] - qq_df['x']
            
            yuplim = np.ceil(10 * np.sqrt(1 / len(qq_df['y'])) * 10) / 10
            yl = [-2, 2]
            xl = [-3, 3]
            
            level = 0.99
            z = np.sort(np.concatenate((np.arange(xl[0] - 1, xl[1] + 1, 0.05), qq_df['x'])))
            p = norm.cdf(z)
            se = (1 / norm.pdf(z)) * (np.sqrt(p * (1 - p) / len(qq_df['y'])))
            low = norm.ppf((1 - level) / 2) * se
            high = -low
            CI = pd.DataFrame({'x': z, 'ymin': low, 'ymax': high})
            CI_aux = pd.merge(CI[CI['x'].isin(qq_df['x'])], qq_df, on='x')
            CI_aux = CI_aux[(CI_aux['y'] < CI_aux['ymin']) | (CI_aux['y'] > CI_aux['ymax'])]
            CI_aux_2 = res[res['QA'].isin(CI_aux['yy'])][['QA', 'ID']].rename(columns={'QA': 'yy'})
            CI_aux = pd.merge(CI_aux, CI_aux_2, on='yy')
            
            X = qq_df[['x']]
            y = qq_df['y']
            reg = LinearRegression().fit(np.hstack((X, X**2, X**3)), y)
            cx = np.linspace(qq_df['x'].min(), qq_df['x'].max(), 50)
            cy = reg.predict(np.hstack((cx.reshape(-1, 1), cx.reshape(-1, 1)**2, cx.reshape(-1, 1)**3)))
            Spline = pd.DataFrame({'x': cx, 'y': cy})
            
            fig, ax = plt.subplots()
            ax.fill_between(CI['x'], CI['ymin'], CI['ymax'], color='black', alpha=0.1)
            ax.plot(Spline['x'], Spline['y'], color='red', linewidth=0.25)
            ax.scatter(qq_df['x'], qq_df['y'], s=0.5)
            ax.axvline(x=0, linestyle='dashed')
            ax.axhline(y=0, linestyle='dashed')
            #for i in range(len(CI_aux)):
            #    ax.text(CI_aux.iloc[i]['x'], CI_aux.iloc[i]['y'], str(CI_aux.iloc[i]['ID']), fontsize=8)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_xlabel('Quantis teóricos N(0,1)')
            ax.set_ylabel('Desvio')
            ax.grid(True)
            st.pyplot(fig)


        # Função para calcular os resíduos Cox-Snell
        def plot_residuos_cox_snell(dados_residual):
            res = pd.DataFrame({
                'CS': dados_residual['res_cox_snell'],
                'C': dados_residual['censura'],
                'MA': dados_residual['res_martingal'],
                'DE': dados_residual['res_deviance'],
                'QA': dados_residual['res_quantilico']
            }).reset_index().rename(columns={'index': 'ID'})

            kmf = KaplanMeierFitter()
            kmf.fit(durations=res['CS'], event_observed=res['C'])

            km_estimates = kmf.survival_function_
            km_conf_int = kmf.confidence_interval_

            fig, ax = plt.subplots()
            ax.step(km_estimates.index, km_estimates['KM_estimate'], where="post", linewidth=0.75)
            ax.fill_between(km_conf_int.index, km_conf_int['KM_estimate_lower_0.95'], km_conf_int['KM_estimate_upper_0.95'], step="post", alpha=0.15)
            
            # Correção da linha vermelha, usando valores ordenados e a curva de sobrevivência esperada
            sorted_cs = np.sort(res['CS'])
            expected_survival = np.exp(-sorted_cs)
            ax.plot(sorted_cs, expected_survival, color='red', linewidth=0.75)
            
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Resíduo de Cox-Snell')
            ax.set_ylabel(r'$\hat{S}$(Cox-Snell)')
            ax.grid(True)
            st.pyplot(fig)


        # Função para plotar os resíduos de martingal
        def plot_residuos_martingal(dados_residual):
            res = pd.DataFrame({
                'ID': np.arange(len(dados_residual)),
                'CS': dados_residual['res_cox_snell'],
                'C': dados_residual['censura'],
                'MA': dados_residual['res_martingal'],
                'DE': dados_residual['res_deviance'],
                'QA': dados_residual['res_quantilico']
            })

            y_range = np.array([-1, 1]) * np.ceil(np.abs(res['MA']).max())

            fig, ax = plt.subplots()
            ax.vlines(x=res['ID'], ymin=0, ymax=res['MA'], color='black', linewidth=0.5)
            ax.scatter(res['ID'], res['MA'], c=res['C'], cmap='viridis', marker='o', s=10)
            ax.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)
            
            ax.set_ylim(y_range)
            ax.set_xlabel('ID')
            ax.set_ylabel(r'$\widehat{m}$')
            ax.grid(True)
            ax.set_title('Resíduos de Martingal')
            ax.set_xticks([])  # Remover rótulos do eixo x
            st.pyplot(fig)


        # Função para plotar os resíduos de Deviance
        def plot_residuos_deviance(dados_residual):
            res = pd.DataFrame({
                'ID': np.arange(len(dados_residual)),
                'CS': dados_residual['res_cox_snell'],
                'C': dados_residual['censura'],
                'MA': dados_residual['res_martingal'],
                'DE': dados_residual['res_deviance'],
                'QA': dados_residual['res_quantilico']
            })

            y_range = np.array([-1, 1]) * np.ceil(np.abs(res['DE']).max())

            fig, ax = plt.subplots()
            ax.vlines(x=res['ID'], ymin=0, ymax=res['DE'], color='black', linewidth=0.5)
            ax.fill_between(np.arange(len(res)), -2, 2, color='black', alpha=0.05)
            ax.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)
            
            ax.set_ylim(y_range)
            ax.set_xlabel('ID')
            ax.set_ylabel(r'$\widehat{d}$')
            ax.grid(True)
            ax.set_title('Resíduos de Deviance')
            ax.set_xticks([])  # Remover rótulos do eixo x
            st.pyplot(fig)

        
        # Função para calcular os dados de previsão
        def calcular_dados_prev(dados_tratados, resumo):
            t_n1 = np.linspace(0, 500, 300)
            aceleracao_unique = dados_tratados['aceleracao'].unique()
            tn = dados_tratados['t'].max()
            
            fa_values = np.exp(resumo.loc[2, 'Estimativa'] + resumo.loc[3, 'Estimativa'] * aceleracao_unique)
            
            dadosprev = pd.DataFrame()
            
            for fa, acel in zip(fa_values, aceleracao_unique):
                H_AFT_tn = H_AFT(tn, resumo.loc[0, 'Estimativa'], resumo.loc[1, 'Estimativa'], fa)
                H_AFT_tn_t_n1 = H_AFT(tn + t_n1, resumo.loc[0, 'Estimativa'], resumo.loc[1, 'Estimativa'], fa)
                R_est = np.exp(H_AFT_tn - H_AFT_tn_t_n1)
                
                temp_df = pd.DataFrame({
                    't_n+1': t_n1,
                    'aceleracao': acel,
                    'R_est': R_est
                })
                
                dadosprev = pd.concat([dadosprev, temp_df], ignore_index=True)
            
            return dadosprev

        # Função para plotar a previsão
        def plot_previsao(dadosprev):
            fig, ax = plt.subplots()
            unique_aceleracoes = dadosprev['aceleracao'].unique()
            color_map = plt.get_cmap('tab10', len(unique_aceleracoes))

            for idx, label in enumerate(unique_aceleracoes):
                df_group = dadosprev[dadosprev['aceleracao'] == label]
                ax.plot(df_group['t_n+1'], df_group['R_est'], label=f'Aceleração {label}', color=color_map(idx))

            ax.set_xlabel('Funcionamento após o último tempo de falha')
            ax.set_ylabel(r'$\hat{R}(t)$')
            ax.legend(title="Nível de aceleração", loc='best')
            ax.grid(True)
            st.pyplot(fig)
        if 'dadostratados' in st.session_state and 'resumo' in st.session_state:
            dadostratados = st.session_state['dadostratados']
            resumo = st.session_state['resumo']

            dados_residual = calcular_dados_residuais(dadostratados, resumo)
            st.session_state['dados_residual'] = dados_residual

            dadosprev = calcular_dados_prev(dadostratados, resumo)
            st.session_state['dadosprev'] = dadosprev

            st.header("Gráficos de diagnóstico e previsão")

# ------------------ Resíduos quantílicos ------------------------------------------
            st.subheader("Resíduos Quantílicos")
            plot_residuos_quantilicos(dados_residual)
# ------------------ Resíduos Cox-Snell --------------------------------------------
            st.subheader("Resíduos Cox-Snell")
            plot_residuos_cox_snell(dados_residual)
# ------------------ Resíduos Martingal --------------------------------------------
            st.subheader("Resíduos de Martingal")
            plot_residuos_martingal(dados_residual)
# ------------------ Resíduos Deviance ---------------------------------------------
            st.subheader("Resíduos de Deviance")
            plot_residuos_deviance(dados_residual)
# ------------------ Previsão ------------------------------------------------------
            st.subheader("Previsão")

            plot_previsao(dadosprev)
