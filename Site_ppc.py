import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Guia de Processos - ENGF79", layout="centered")

# Inicialização do Estado de Navegação
if 'node' not in st.session_state:
    st.session_state.node = 'inicio'

# Estado para a otimização automática na Modelagem Empírica
if 'opt_res' not in st.session_state:
    st.session_state.opt_res = {"done": False, "K": 0.0, "tau": 0.0, "mse": 0.0}

# ==============================================================================
# NÓ 1: INÍCIO
# ==============================================================================
if st.session_state.node == 'inicio':
    st.title("🎓 Guia Interativo de PPC")
    st.write("Navegue pelas perguntas para encontrar o caminho correto para o seu problema. Produto tecnológico feito por Nicole.")
    st.header("Qual é o seu principal objetivo agora?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Criar um Modelo Matemático", use_container_width=True):
            st.session_state.node = 'criar_modelo'
            st.rerun()
    with col2:
        if st.button(" Analisar Modelo Existente", use_container_width=True):
            st.session_state.node = 'analisar_modelo'
            st.rerun()

# ==============================================================================
# NÓ 2: ESCOLHA DA ABORDAGEM
# ==============================================================================
elif st.session_state.node == 'criar_modelo':
    st.header("Qual é o seu ponto de partida para a modelagem?")
    st.info("A escolha da abordagem depende do conhecimento prévio que possui sobre o sistema.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚪ Modelagem Teórica\n**Princípios Fundamentais (Caixa Branca)**")
        if st.button("Aprender o passo a passo", use_container_width=True):
            st.session_state.node = 'modelo_teorico'
            st.rerun()
            
    with col2:
        st.markdown("### ⚫ Modelagem Empírica\n**Identificação de Sistemas (Caixa Preta)**")
        if st.button("Usar dados experimentais", use_container_width=True):
            st.session_state.node = 'modelo_empirico'
            st.rerun()
        
    if st.button("⬅️ Voltar ao Início", use_container_width=True):
        st.session_state.node = 'inicio'
        st.rerun()
        
elif st.session_state.node == 'analisar_modelo':
        st.header("Análise Dinâmica de Sistemas")
        st.markdown("""
        Esta seção aprofunda os conceitos de **Linearização** (aproximação de modelos reais) e **Estabilidade** (análise de autovalores e pólos).
        """)
        
        # Abas para separar os tópicos complexos
        tab_lin, tab_estab, tab_resp = st.tabs(["📏 Linearização", "💥 Estabilidade (Matriz A)", "📈 Tipos de Resposta"])

        # --- ABA 1: LINEARIZAÇÃO ---
        with tab_lin:
            st.subheader("Linearização: A Ponte entre o Real e o Matemático")
            
            # Vídeo didático restaurado
            st.video("https://youtu.be/AwduhbnWLc8?si=fk12sKdD7qnn4PyE")
            st.caption("Vídeo Sugerido: Linearização e Laplace - Produto tecnológico de Matheus Marinho.")
            
            st.markdown("""
            ### 1. Conceito: O Mundo é Não-Linear
            Na engenharia, quase todos os sistemas reais são **Não-Lineares**.
            * **Exemplo:** A vazão em uma válvula não dobra se você dobrar a abertura; ela segue uma raiz quadrada ($q \propto \sqrt{h}$).
            * **Problema:** Ferramentas poderosas como **Laplace** e **Função de Transferência** só funcionam para sistemas lineares ($y = ax + b$).
            
            **A Solução:** Aproximar a curva real por uma **Reta Tangente** em uma pequena região ao redor de onde o sistema opera (Ponto de Operação).
            """)

            # EXPANSÃO MATEMÁTICA DETALHADA
            with st.expander("📚 FUNDAMENTAÇÃO MATEMÁTICA COMPLETA: Da Não-Linearidade à Função de Transferência"):
                
                st.markdown("## Parte 1: Matemática da Linearização")
                
                # 1.1 Série de Taylor
                st.markdown("### 1.1 Série de Taylor (Expansão em torno de um ponto)")
                st.markdown("Qualquer função $f(x)$ infinitamente diferenciável pode ser expressa como:")
                st.latex(r"""
                f(x) = f(\bar{x}) + f'(\bar{x})(x-\bar{x}) + \frac{f''(\bar{x})}{2!}(x-\bar{x})^2 + \frac{f'''(\bar{x})}{3!}(x-\bar{x})^3 + \cdots
                """)
                
                col_taylor1, col_taylor2 = st.columns(2)
                with col_taylor1:
                    st.markdown("**Aproximação Linear (1ª ordem):**")
                    st.latex(r"f(x) \approx f(\bar{x}) + f'(\bar{x})(x-\bar{x})")
                    st.markdown("**Erro de truncamento:**")
                    st.latex(r"R_1(x) = \frac{f''(\xi)}{2!}(x-\bar{x})^2, \quad \xi \in [\bar{x}, x]")
                
                with col_taylor2:
                    st.markdown("**Interpretação Geométrica:**")
                    st.markdown("""
                    - $f\\bar{x}$: Valor da função no ponto de operação
                    - $f'\\bar{x}$: Inclinação da reta tangente
                    - $x-\\bar{x}$: Distância do ponto de operação
                    """)
                
                # 1.2 Sistemas Multivariáveis
                st.markdown("### 1.2 Linearização de Sistemas Multivariáveis")
                st.markdown("Para uma função $f(x_1, x_2, \dots, x_n)$:")
                st.latex(r"""
                f(\mathbf{x}) \approx f(\bar{\mathbf{x}}) + \sum_{i=1}^n \left.\frac{\partial f}{\partial x_i}\right|_{\bar{\mathbf{x}}} (x_i - \bar{x}_i)
                """)
                st.markdown("Ou em forma vetorial:")
                st.latex(r"""
                f(\mathbf{x}) \approx f(\bar{\mathbf{x}}) + \nabla f(\bar{\mathbf{x}})^T (\mathbf{x} - \bar{\mathbf{x}})
                """)
                
                st.divider()
                
                st.markdown("## Parte 2: Transformada de Laplace e Domínios")
                
                st.markdown("### 2.1 Definição da Transformada de Laplace")
                st.latex(r"""
                \mathcal{L}\{f(t)\} = F(s) = \int_{0}^{\infty} f(t)e^{-st}dt, \quad s = \sigma + j\omega
                """)
                
                # Tabela de transformadas importantes
                st.markdown("### 2.2 Transformadas Essenciais para Sistemas Dinâmicos")
                
                col_tf1, col_tf2 = st.columns(2)
                
                with col_tf1:
                    st.markdown("""
                    | **Domínio do Tempo** | **Domínio de Laplace** |
                    |----------------------|------------------------|
                    | $f(t) = K$ | $F(s) = \\frac{K}{s}$ |
                    | $f(t) = e^{-at}$ | $F(s) = \\frac{1}{s+a}$ |
                    | $f(t) = \\sin(\\omega t)$ | $F(s) = \\frac{\\omega}{s^2+\\omega^2}$ |
                    | $f(t) = \\cos(\\omega t)$ | $F(s) = \\frac{s}{s^2+\\omega^2}$ |
                    """)
                
                with col_tf2:
                    st.markdown("""
                    | **Domínio do Tempo** | **Domínio de Laplace** |
                    |----------------------|------------------------|
                    | $\\frac{df(t)}{dt}$ | $sF(s) - f(0^+)$ |
                    | $\\int_0^t f(\\tau)d\\tau$ | $\\frac{F(s)}{s}$ |
                    | $f(t-\\tau)u(t-\\tau)$ | $e^{-s\\tau}F(s)$ |
                    """)
                    
                st.markdown("### 2.3 Propriedades para Equações Diferenciais")
                
                st.markdown("**Com condições iniciais nulas:**")
                st.latex(r"""
                \begin{aligned}
                \mathcal{L}\left\{\frac{d^n y}{dt^n}\right\} &= s^n Y(s) \\
                \mathcal{L}\left\{\int y(t)dt\right\} &= \frac{Y(s)}{s}
                \end{aligned}
                """)
                
                st.divider()
                
                st.markdown("## Parte 3: De Sistemas Não-Lineares à Função de Transferência")
                
                st.markdown("### 3.1 Processo Geral de Linearização")
                
                st.markdown("""
                **Passo 1:** Identificar as variáveis de estado e entrada
                
                **Passo 2:** Estabelecer ponto de operação $\\bar{x}, \\bar{u}$
                
                **Passo 3:** Definir variáveis de desvio:
                """)
                st.latex(r"""
                \begin{aligned}
                \hat{x}(t) &= x(t) - \bar{x} \\
                \hat{u}(t) &= u(t) - \bar{u}
                \end{aligned}
                """)
                
                st.markdown("**Passo 4:** Aplicar expansão de Taylor à EDO não-linear")
                
                st.markdown("**Passo 5:** Aplicar Transformada de Laplace")
                
                st.markdown("### 3.2 Exemplo Teórico Completo")
                
                st.markdown("Considere um sistema não-linear genérico:")
                st.latex(r"""
                \frac{dx}{dt} = f(x, u)
                """)
                
                st.markdown("**Expansão em torno de $\\bar{x}, {\\bar}{u}$:**")
                st.latex(r"""
                \begin{aligned}
                f(x, u) &\approx f(\bar{x}, \bar{u}) + \left.\frac{\partial f}{\partial x}\right|_{\bar{x},\bar{u}}(x-\bar{x}) + \left.\frac{\partial f}{\partial u}\right|_{\bar{x},\bar{u}}(u-\bar{u}) \\
                &= \underbrace{f(\bar{x}, \bar{u})}_{=0 \text{ em equilíbrio}} + A\hat{x} + B\hat{u}
                \end{aligned}
                """)
                
                st.markdown("**EDO linearizada:**")
                st.latex(r"""
                \frac{d\hat{x}}{dt} = A\hat{x} + B\hat{u}
                """)
                
                st.markdown("**Aplicando Laplace:**")
                st.latex(r"""
                \begin{aligned}
                s\hat{X}(s) - \hat{x}(0) &= A\hat{X}(s) + B\hat{U}(s) \\
                (s - A)\hat{X}(s) &= B\hat{U}(s) \quad \text{(condições iniciais nulas)} \\
                \frac{\hat{X}(s)}{\hat{U}(s)} &= \frac{B}{s - A}
                \end{aligned}
                """)
                
                st.markdown("### 3.3 Formas Canônicas de Sistemas Lineares")
                
                col_forms1, col_forms2 = st.columns(2)
                
                with col_forms1:
                    st.markdown("**Sistema de 1ª Ordem:**")
                    st.latex(r"""
                    \begin{aligned}
                    \text{EDO: } & \tau\frac{dy}{dt} + y = Ku(t) \\
                    \text{FT: } & G(s) = \frac{K}{\tau s + 1}
                    \end{aligned}
                    """)
                    st.markdown("- ${\tau}$: Constante de tempo")
                    st.markdown("- $K$: Ganho estático")
                
                with col_forms2:
                    st.markdown("**Sistema de 2ª Ordem:**")
                    st.latex(r"""
                    \begin{aligned}
                    \text{EDO: } & \frac{d^2y}{dt^2} + 2{\zeta}{\omega_n}{\frac{dy}{dt}} + \omega_n^2 y = K\omega_n^2 u(t) \\
                    \text{FT: } & G(s) = \frac{K\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}
                    \end{aligned}
                    """)
                    st.markdown("- $\omega_n$: Frequência natural")
                    st.markdown("- $\zeta$: Coeficiente de amortecimento")
                
                st.divider()
                
                st.markdown("## Parte 4: Espaço de Estados (Forma Matricial)")
                
                st.markdown("### 4.1 Representação Geral")
                st.latex(r"""
                \begin{aligned}
                \frac{d\mathbf{x}(t)}{dt} &= A\mathbf{x}(t) + B\mathbf{u}(t) \\
                \mathbf{y}(t) &= C\mathbf{x}(t) + D\mathbf{u}(t)
                \end{aligned}
                """)
                
                st.markdown("Onde:")
                st.latex(r"""
                \begin{aligned}
                \mathbf{x} &\in \mathbb{R}^n \quad \text{(vetor de estados)} \\
                \mathbf{u} &\in \mathbb{R}^m \quad \text{(vetor de entradas)} \\
                \mathbf{y} &\in \mathbb{R}^p \quad \text{(vetor de saídas)} \\
                A &\in \mathbb{R}^{n\times n} \quad \text{(matriz do sistema)} \\
                B &\in \mathbb{R}^{n\times m} \quad \text{(matriz de entrada)} \\
                C &\in \mathbb{R}^{p\times n} \quad \text{(matriz de saída)} \\
                D &\in \mathbb{R}^{p\times m} \quad \text{(matriz de transmissão direta)}
                \end{aligned}
                """)
                
                st.markdown("### 4.2 Transformada de Laplace na Forma Matricial")
                st.latex(r"""
                \begin{aligned}
                \mathcal{L}\left\{\frac{d\mathbf{x}}{dt}\right\} &= \mathcal{L}\{A\mathbf{x} + B\mathbf{u}\} \\
                s\mathbf{X}(s) - \mathbf{x}(0) &= A\mathbf{X}(s) + B\mathbf{U}(s) \\
                (sI - A)\mathbf{X}(s) &= \mathbf{x}(0) + B\mathbf{U}(s) \\
                \mathbf{X}(s) &= (sI - A)^{-1}[\mathbf{x}(0) + B\mathbf{U}(s)]
                \end{aligned}
                """)
                
                st.markdown("**Função de Transferência (condições iniciais nulas):**")
                st.latex(r"""
                G(s) = \frac{\mathbf{Y}(s)}{\mathbf{U}(s)} = C(sI - A)^{-1}B + D
                """)
                
                st.markdown("**Polos do sistema:** Raízes de $\det(sI - A) = 0$")

            # EXPANSÃO DO EXISTENTE
            with st.expander("📚 A Matemática: De Taylor a Laplace (Versão Aplicada)"):
                
                st.markdown("#### 1. Série de Taylor Aplicada à Dinâmica")
                st.markdown("Para um sistema dinâmico não-linear $\\dot{x} = f(x, u)$:")
                st.latex(r"""
                f(x, u) \approx f(\bar{x}, \bar{u}) + \left.\frac{\partial f}{\partial x}\right|_{\bar{x},\bar{u}}(x-\bar{x}) + \left.\frac{\partial f}{\partial u}\right|_{\bar{x},\bar{u}}(u-\bar{u})
                """)
                
                st.markdown("#### 2. Variáveis de Desvio com Significado Físico")
                st.markdown("""
                **Por que usar variáveis de desvio?**
                1. O ponto de equilíbrio $f(\bar{x}, \bar{u}) = 0$ se cancela
                2. Trabalhamos apenas com variações
                3. Matematicamente mais simples
                4. Fisicamente mais significativo
                """)
                st.latex(r"""
                \begin{aligned}
                \hat{x} &= x - \bar{x} \\
                \hat{u} &= u - \bar{u} \\
                \dot{\hat{x}} &= \dot{x} \quad \text{(pois } \dot{\bar{x}} = 0\text{)}
                \end{aligned}
                """)
                
                st.markdown("#### 3. Da EDO Linear à Função de Transferência")
                st.markdown("**EDO linear resultante:**")
                st.latex(r"""
                \dot{\hat{x}} = A\hat{x} + B\hat{u}, \quad \text{onde } A = \left.\frac{\partial f}{\partial x}\right|_{\bar{x},\bar{u}}, \quad B = \left.\frac{\partial f}{\partial u}\right|_{\bar{x},\bar{u}}
                """)
                
                st.markdown("**Aplicando Laplace:**")
                st.latex(r"""
                \begin{aligned}
                s\hat{X}(s) &= A\hat{X}(s) + B\hat{U}(s) \\
                (s - A)\hat{X}(s) &= B\hat{U}(s) \\
                \frac{\hat{X}(s)}{\hat{U}(s)} &= \frac{B}{s - A}
                \end{aligned}
                """)
                
                st.markdown("**Para sistemas de ordem superior:**")
                st.latex(r"""
                G(s) = \frac{\hat{Y}(s)}{\hat{U}(s)} = \frac{b_m s^m + \cdots + b_1 s + b_0}{a_n s^n + \cdots + a_1 s + a_0}
                """)

            st.divider()

            st.subheader("2. Estudo de Caso: Tanque de Nível (Dinâmico)")
            # TÍTULO PRINCIPAL
            st.markdown("### Simulação: Tanque de Nível - Linearização e Validação")
            
            # CONFIGURAÇÃO FIXA
            A_tanque = 2.0
            k_valve = 1.5
            h_bar = 4.0  # Ponto de operação FIXO
            q_bar = k_valve * np.sqrt(h_bar)
            
            # MOSTRAR INFORMAÇÕES DE EQUILÍBRIO
            st.markdown("#### Informações do Sistema")
            col_eq1, col_eq2, col_eq3 = st.columns(3)
            
            with col_eq1:
                st.markdown("**Parâmetros fixos:**")
                st.write(f"Área do tanque: A = {A_tanque} m²")
                st.write(f"Coeficiente da válvula: k = {k_valve}")
            
            with col_eq2:
                st.markdown("**Ponto de operação:**")
                st.write(f"Altura: h̄ = {h_bar} m")
                st.write(f"Vazão: q̄ = {q_bar:.3f} m³/s")
            
            with col_eq3:
                st.markdown("**Equilíbrio:**")
                st.latex(r"q_{in} = q_{out} = k\sqrt{h}")
                st.latex(fr"{q_bar:.3f} = {k_valve}\times\sqrt{{{h_bar}}}")
            
            st.markdown("---")
            
            # SEÇÃO: TEORIA DOS MÉTODOS
            st.markdown("#### Métodos para Obter os Parâmetros do Modelo Linear")
            
            metodo = st.radio(
                "Escolha o método de obtenção dos parâmetros:",
                ["Método Teórico (análise matemática)", "Método Gráfico (resposta experimental)"],
                horizontal=True
            )
            
            if metodo == "Método Teórico (análise matemática)":
                with st.expander("📝 Ver cálculo teórico detalhado"):
                    st.markdown("**Equação não-linear do tanque:**")
                    st.latex(r"A \frac{dh}{dt} = q_{in} - k\sqrt{h}")
                    
                    st.markdown("**1. Linearização pelo ponto de operação:**")
                    st.latex(r"\text{Ponto de operação: } h = \bar{h}, \quad q_{in} = \bar{q}_{in}")
                    st.latex(r"\text{Onde: } \bar{q}_{in} = k\sqrt{\bar{h}}")
                    
                    st.markdown("**2. Variáveis de desvio:**")
                    st.latex(r"\hat{h} = h - \bar{h}, \quad \hat{q} = q_{in} - \bar{q}_{in}")
                    
                    st.markdown("**3. Expansão em série de Taylor de $\sqrt{h}$:**")
                    st.latex(r"""
                    \begin{aligned}
                    \sqrt{h} &= \sqrt{\bar{h} + \hat{h}} \\
                    &\approx \sqrt{\bar{h}} + \left.\frac{d}{dh}\sqrt{h}\right|_{\bar{h}} \hat{h} \\
                    &= \sqrt{\bar{h}} + \frac{1}{2\sqrt{\bar{h}}}\hat{h}
                    \end{aligned}
                    """)
                    
                    st.markdown("**4. Substituição na equação:**")
                    st.latex(r"""
                    \begin{aligned}
                    A \frac{d(\bar{h} + \hat{h})}{dt} &= (\bar{q}_{in} + \hat{q}) - k\left(\sqrt{\bar{h}} + \frac{1}{2\sqrt{\bar{h}}}\hat{h}\right) \\
                    A \frac{d\hat{h}}{dt} &= \hat{q} - \frac{k}{2\sqrt{\bar{h}}}\hat{h}
                    \end{aligned}
                    """)
                    
                    st.markdown("**5. Reorganizando na forma padrão:**")
                    st.latex(r"""
                    \frac{2A\sqrt{\bar{h}}}{k} \frac{d\hat{h}}{dt} + \hat{h} = \frac{2\sqrt{\bar{h}}}{k} \hat{q}
                    """)
                    
                    st.markdown("**6. Identificando os parâmetros:**")
                    st.latex(r"""
                    \begin{aligned}
                    \tau &= \frac{2A\sqrt{\bar{h}}}{k} \quad \text{(constante de tempo)} \\
                    K_p &= \frac{2\sqrt{\bar{h}}}{k} \quad \text{(ganho estático)}
                    \end{aligned}
                    """)
                    
                    st.markdown("**7. Substituindo valores numéricos:**")
                    st.latex(fr"""
                    \begin{{aligned}}
                    \tau &= \frac{{2 \times {A_tanque} \times \sqrt{{{h_bar}}}}}{{{k_valve}}}
                    = \frac{{2 \times {A_tanque} \times {np.sqrt(h_bar):.2f}}}{{{k_valve}}} 
                    = {2*A_tanque*np.sqrt(h_bar)/k_valve:.2f}\ \text{{s}} \\
                    K_p &= \frac{{2 \times \sqrt{{{h_bar}}}}}{{{k_valve}}} 
                    = \frac{{2 \times {np.sqrt(h_bar):.2f}}}{{{k_valve}}}
                    = {2*np.sqrt(h_bar)/k_valve:.2f}\ \text{{m/(m³/s)}}
                    \end{{aligned}}
                    """)
            
            else:  # Método Gráfico
                with st.expander("📝 Ver procedimento do método gráfico"):
                    st.markdown("**Procedimento experimental:**")
                    
                    st.markdown("1. **Aplicar um degrau pequeno** na entrada:")
                    st.latex(r"\Delta q = 0.1\ \text{m³/s} \quad \text{(perturbação pequena)}")
                    
                    st.markdown("2. **Medir a resposta** do sistema (altura h(t) ao longo do tempo)")
                    
                    st.markdown("3. **Do gráfico da resposta, obter:**")
                    
                    col_graf1, col_graf2 = st.columns(2)
                    
                    with col_graf1:
                        st.markdown("**Ganho Kp:**")
                        st.latex(r"K_p = \frac{\Delta h_{\infty}}{\Delta q}")
                        st.markdown("Onde:")
                        st.markdown("- $\Delta h_{\infty}$ = variação final da altura")
                        st.markdown("- $\Delta q$ = tamanho do degrau aplicado")
                    
                    with col_graf2:
                        st.markdown("**Constante de tempo τ:**")
                        st.latex(r"\tau = t_{63.2\%}")
                        st.markdown("Onde:")
                        st.markdown("- $t_{63.2\%}$ = tempo para atingir 63.2% da variação total")
                        st.markdown("- Ou: tempo onde $h(t) = h_0 + 0.632 \times \Delta h_{\infty}$")
                    
                    st.markdown("**4. Função de transferência resultante:**")
                    st.latex(r"G(s) = \frac{K_p}{\tau s + 1}")
            
            # CALCULAR PARÂMETROS CONFORME MÉTODO ESCOLHIDO
            if metodo == "Método Teórico (análise matemática)":
                # Cálculo teórico
                R_hid = (2 * np.sqrt(h_bar)) / k_valve
                tau = A_tanque * R_hid
                Kp = R_hid
                
                st.markdown("#### Parâmetros Calculados (Método Teórico)")
                col_param1, col_param2 = st.columns(2)
                
                with col_param1:
                    st.latex(fr"\tau = \frac{{2A\sqrt{{\bar{{h}}}}}}{{k}} = {tau:.2f}\ \text{{s}}")
                
                with col_param2:
                    st.latex(fr"K_p = \frac{{2\sqrt{{\bar{{h}}}}}}{{k}} = {Kp:.2f}\ \text{{m/(m³/s)}}")
            
            else:
                # Método gráfico - identificar da resposta
                step_ident = 0.1  # Degrau pequeno para identificação
                
                # Simulação de identificação
                t_ident = np.arange(0, 30, 0.1)
                h_ident = np.zeros_like(t_ident)
                h_ident[0] = h_bar
                
                for i in range(1, len(t_ident)):
                    if t_ident[i] < 5:
                        q_ident = q_bar
                    else:
                        q_ident = q_bar + step_ident
                    
                    dh_dt = (q_ident - k_valve * np.sqrt(max(h_ident[i-1], 0.01))) / A_tanque
                    h_ident[i] = max(h_ident[i-1] + dh_dt * 0.1, 0.01)
                
                # Calcular Kp do gráfico
                h_final_ident = h_ident[-1]
                delta_h_ident = h_final_ident - h_bar
                Kp = delta_h_ident / step_ident
                
                # Calcular τ do gráfico
                h_target = h_bar + 0.632 * delta_h_ident
                idx_tau = np.argmax(h_ident >= h_target)
                tau = t_ident[idx_tau] - 5 if idx_tau > 0 else 0
                
                # Valores teóricos para comparação
                R_hid_teorico = (2 * np.sqrt(h_bar)) / k_valve
                tau_teorico = A_tanque * R_hid_teorico
                Kp_teorico = R_hid_teorico
                
                st.markdown("#### Parâmetros Identificados (Método Gráfico)")
                
                col_iden1, col_iden2, col_iden3 = st.columns(3)
                
                with col_iden1:
                    st.metric("τ identificado", f"{tau:.2f} s")
                    st.caption(f"Teórico: {tau_teorico:.2f} s")
                
                with col_iden2:
                    st.metric("Kp identificado", f"{Kp:.2f} m/(m³/s)")
                    st.caption(f"Teórico: {Kp_teorico:.2f} m/(m³/s)")
                
                with col_iden3:
                    st.metric("Erro na identificação", 
                             f"{(abs(tau-tau_teorico)/tau_teorico*100):.1f}%")
            
            st.markdown("---")
            
            # SEÇÃO: FUNÇÃO DE TRANSFERÊNCIA
            st.markdown("#### Modelo Linear - Função de Transferência")
            
            col_ft1, col_ft2 = st.columns([1, 2])
            
            with col_ft1:
                st.markdown("**Forma geral:**")
                st.latex(r"G(s) = \frac{K_p}{\tau s + 1}")
            
            with col_ft2:
                st.markdown("**Com valores numéricos:**")
                st.latex(fr"G(s) = \frac{{{Kp:.2f}}}{{{tau:.2f}\ s + 1}}")
                st.latex(r"\delta H(s) = G(s) \cdot \delta Q(s)")
            
            st.markdown("---")
            
            # SEÇÃO: CONFIGURAÇÃO DO TESTE DE VALIDAÇÃO
            st.markdown("#### Teste de Validação do Modelo")
            
            step_size = st.slider(
                "**Tamanho do degrau de teste (Δq):**",
                -2.0, 2.0, 0.5, 0.1,
                help="Variação na vazão de entrada para testar o modelo"
            )
            
            st.markdown(f"**Configuração do teste:**")
            st.markdown(f"- Ponto de operação: h̄ = {h_bar} m, q̄ = {q_bar:.3f} m³/s")
            st.markdown(f"- Degrau aplicado: Δq = {step_size} m³/s")
            st.markdown(f"- Vazão total: q = {q_bar:.3f} {'+' if step_size >= 0 else ''}{step_size} = {q_bar + step_size:.3f} m³/s")
            
            st.markdown("---")
            
            # SIMULAÇÃO E GRÁFICO
            st.markdown("#### Resposta ao Degrau - Comparação Modelo vs Real")
            
            # Parâmetros da simulação
            t_max = 50
            dt = 0.05
            time = np.arange(0, t_max, dt)
            steps = len(time)
            
            # SIMULAR RESPOSTAS
            h_nonlinear = np.zeros(steps)
            h_linear_dev = np.zeros(steps)
            h_nonlinear[0] = h_bar
            h_linear_dev[0] = 0.0
            
            for i in range(1, steps):
                if time[i] < 5.0:
                    q_in_atual = q_bar
                    delta_q = 0.0
                else:
                    q_in_atual = q_bar + step_size
                    delta_q = step_size
                
                # Modelo REAL (Não-Linear)
                dh_dt = (q_in_atual - k_valve * np.sqrt(max(h_nonlinear[i-1], 0.01))) / A_tanque
                h_nonlinear[i] = max(h_nonlinear[i-1] + dh_dt * dt, 0.01)
                
                # Modelo LINEAR (Aproximado)
                d_deltah_dt = (Kp * delta_q - h_linear_dev[i-1]) / tau
                h_linear_dev[i] = h_linear_dev[i-1] + d_deltah_dt * dt
            
            # Converter desvio para valor absoluto
            h_linear_abs = h_linear_dev + h_bar
            
            # CRIAR GRÁFICO PRINCIPAL
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Plotar respostas
            ax.plot(time, h_nonlinear, 'b-', linewidth=2.5, label='Sistema Real (Não-Linear)')
            ax.plot(time, h_linear_abs, 'r--', linewidth=2, label='Modelo Linearizado')
            
            # Linhas de referência
            ax.axhline(y=h_bar, color='black', linestyle=':', alpha=0.7, 
                      linewidth=2, label=f'Ponto de operação: h̄ = {h_bar} m')
            ax.axvline(x=5, color='gray', linestyle='--', alpha=0.7, 
                      linewidth=1.5, label=f'Degrau aplicado: Δq = {step_size} m³/s')
            
            # Se método gráfico, adicionar marcações
            if metodo == "Método Gráfico (resposta experimental)":
                # Linha de 63.2%
                h_final = h_nonlinear[-1]
                delta_h = h_final - h_bar
                h_63 = h_bar + 0.632 * delta_h
                
                ax.axhline(y=h_63, color='green', linestyle='--', alpha=0.6, 
                          linewidth=1.5, label='63.2% da variação total')
                
                # Encontrar e marcar τ
                idx_63 = np.argmax(h_nonlinear >= h_63)
                if idx_63 > 0:
                    t_63 = time[idx_63]
                    ax.axvline(x=t_63, color='green', linestyle='--', alpha=0.6,
                              linewidth=1.5, label=f'τ = {t_63-5:.1f} s (identificado)')
            
            # Configurações do gráfico
            ax.set_xlabel('Tempo (s)', fontsize=11)
            ax.set_ylabel('Altura do nível h(t) (m)', fontsize=11)
            ax.set_title(f'Validação do Modelo Linear | Ponto de operação: h̄ = {h_bar} m', 
                       fontsize=13, pad=10)
            
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, t_max])
            
            st.pyplot(fig)
            
            st.markdown("---")
            
            # SEÇÃO: ANÁLISE DE ERRO
            st.markdown("#### Análise Quantitativa do Erro")
            
            # Calcular várias métricas de erro
            erro_abs = h_nonlinear - h_linear_abs
            erro_final = abs(h_nonlinear[-1] - h_linear_abs[-1])
            erro_max = np.max(np.abs(erro_abs))
            
            # Cálculo do erro relativo percentual
            with st.expander("📝 Como calcular o erro relativo"):
                st.markdown("**Fórmula do erro relativo percentual:**")
                st.latex(r"\text{Erro relativo} = \left|\frac{\text{Valor real} - \text{Valor aproximado}}{\text{Valor real}}\right| \times 100\%")
                
                st.markdown("**No nosso caso:**")
                st.latex(fr"""
                \begin{{aligned}}
                \text{{Erro relativo final}} &= \left|\frac{{h_{{\text{{real}}}} - h_{{\text{{modelo}}}}}}{{h_{{\text{{real}}}}}}\right| \times 100\% \\
                &= \left|\frac{{{h_nonlinear[-1]:.3f} - {h_linear_abs[-1]:.3f}}}{{{h_nonlinear[-1]:.3f}}}\right| \times 100\% \\
                &= {abs((h_nonlinear[-1] - h_linear_abs[-1])/h_nonlinear[-1])*100:.2f}\%
                \end{{aligned}}
                """)
                
                st.markdown("**Interpretação:**")
                st.markdown("- **< 2%**: Excelente aproximação")
                st.markdown("- **2-10%**: Aproximação aceitável")
                st.markdown("- **> 10%**: Aproximação inadequada")
            
            # Métricas de erro
            perc_erro = (erro_final / h_nonlinear[-1]) * 100
            
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                st.metric("Altura final real", f"{h_nonlinear[-1]:.3f} m")
            
            with col_met2:
                st.metric("Altura final modelo", f"{h_linear_abs[-1]:.3f} m")
            
            with col_met3:
                st.metric("Erro absoluto final", f"{erro_final:.4f} m")
            
            with col_met4:
                st.metric("Erro relativo final", f"{perc_erro:.2f}%")
            
            # Erro máximo durante toda a simulação
            col_erro1, col_erro2 = st.columns(2)
            
            with col_erro1:
                st.metric("Erro máximo absoluto", f"{erro_max:.4f} m")
            
            with col_erro2:
                # Tempo onde ocorre o erro máximo
                idx_erro_max = np.argmax(np.abs(erro_abs))
                st.metric("Tempo do erro máximo", f"{time[idx_erro_max]:.1f} s")
            
            st.markdown("---")
            
            # SEÇÃO: CONCLUSÃO DA VALIDAÇÃO
            st.markdown("#### Conclusão da Validação do Modelo")
            
            if perc_erro < 2.0:
                st.success(f"""
                **✅ VALIDAÇÃO BEM-SUCEDIDA - Excelente aproximação!**
                
                O modelo linear representa muito bem o sistema real com apenas **{perc_erro:.1f}%** de erro.
                
                **O que isso significa:**
                1. A linearização foi adequada para esta perturbação
                2. O modelo captura a dinâmica essencial do sistema
                3. Pode ser usado para projeto de controladores lineares
                """)
            elif perc_erro < 10.0:
                st.info(f"""
                **⚠️ VALIDAÇÃO PARCIALMENTE BEM-SUCEDIDA - Aproximação aceitável**
                
                O modelo linear captura a tendência principal com **{perc_erro:.1f}%** de erro.
                
                **Considerações:**
                1. O modelo é útil para análise qualitativa
                2. Pode precisar de ajustes para controle preciso
                3. Válido para pequenas perturbações em torno do ponto de operação
                """)
            else:
                st.error(f"""
                **❌ VALIDAÇÃO FALHOU - Linearização inadequada**
                
                O erro é muito alto: **{perc_erro:.1f}%**.
                
                **Possíveis causas:**
                1. Degrau muito grande em relação ao ponto de operação
                2. Não-linearidade significativa do sistema
                3. Ponto de operação inadequado
                
                **Recomendações:**
                1. Reduza o tamanho do degrau (Δq < 0.5 m³/s)
                2. Considere técnicas de controle não-linear
                3. Valide em diferentes pontos de operação
                """)
            
            st.markdown("---")
            
            # SEÇÃO: RESUMO FINAL
            st.markdown("#### Resumo do Estudo de Caso")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("**Sistema original (não-linear):**")
                st.latex(r"A \frac{dh}{dt} = q_{in} - k\sqrt{h}")
                st.markdown(f"- A = {A_tanque} m²")
                st.markdown(f"- k = {k_valve}")
                st.markdown(f"- Ponto de operação: h̄ = {h_bar} m")
            
            with col_res2:
                st.markdown("**Modelo linearizado:**")
                st.latex(fr"G(s) = \frac{{{Kp:.2f}}}{{{tau:.2f}\ s + 1}}")
                st.markdown(f"- τ = {tau:.2f} s (constante de tempo)")
                st.markdown(f"- Kp = {Kp:.2f} m/(m³/s) (ganho estático)")
                st.markdown(f"- Erro de validação: {perc_erro:.1f}%")
            
            st.markdown("**Aprendizados principais:**")
            st.markdown("""
            1. A linearização via série de Taylor permite aproximar sistemas não-lineares
            2. Os parâmetros τ e Kp podem ser obtidos teoricamente ou experimentalmente
            3. A validade da aproximação depende do tamanho da perturbação
            4. O erro relativo é a métrica mais importante para validação
            5. Modelos lineares são ferramentas poderosas quando usados dentro de sua região de validade
            """)

        # --- ABA 2: ESTABILIDADE E AUTOVALORES ---
        with tab_estab:
            st.subheader("Estabilidade via Espaço de Estados e Autovalores")
            
            # INTRODUÇÃO TEÓRICA
            st.markdown("""
            ### Conceitos Fundamentais
            
            Em sistemas modelados por **espaço de estados**:
            
            $$
            \\begin{aligned}
            \\dot{\\mathbf{x}}(t) &= A\\mathbf{x}(t) + B\\mathbf{u}(t) \\\\
            \\mathbf{y}(t) &= C\\mathbf{x}(t) + D\\mathbf{u}(t)
            \\end{aligned}
            $$
            
            A **estabilidade** é determinada exclusivamente pela matriz $A$:
            
            - **Autovalores** ($\\lambda_i$) são as raízes de $\\det(sI - A) = 0$
            - Cada autovalor $\\lambda_i$ corresponde a um modo natural $e^{\\lambda_i t}$
            - A **parte real** dos autovalores define a estabilidade
            """)
            
            st.divider()
            
            # CRITÉRIOS DE ESTABILIDADE (APENAS TEXTO)
            st.markdown("### Critérios de Estabilidade Baseados nos Autovalores")
            
            col_crit1, col_crit2, col_crit3 = st.columns(3)
            
            with col_crit1:
                st.markdown("""
                #### **Sistema Estável**
                
                **Condição:** Todos os autovalores têm parte real **negativa**
                
                **Comportamento:** 
                - Resposta converge exponencialmente para o equilíbrio
                - Sistema retorna ao repouso após qualquer perturbação
                - Estados tendem a zero quando $t \\to \\infty$
                
                **Exemplo matemático:**
                $\\lambda_1 = -1 + 2j$, $\\lambda_2 = -1 - 2j$
                
                **Modos naturais:** $e^{-t}(\\cos 2t + j\\sin 2t)$
                """)
                
            with col_crit2:
                st.markdown("""
                #### **Sistema Marginalmente Estável**
                
                **Condição:** Autovalores com parte real **zero**, nenhum no SPD
                
                **Comportamento:**
                - Oscilações sustentadas (sem amortecimento)
                - Resposta constante, não converge nem diverge
                - Sistema neutro à perturbações
                
                **Exemplo matemático:**
                $\\lambda_1 = 0 + 3j$, $\\lambda_2 = 0 - 3j$
                
                **Modos naturais:** $\\cos 3t$, $\\sin 3t$
                """)
                
            with col_crit3:
                st.markdown("""
                #### **Sistema Instável**
                
                **Condição:** Pelo menos um autovalor tem parte real **positiva**
                
                **Comportamento:**
                - Resposta diverge exponencialmente com o tempo
                - Sistema não retorna ao equilíbrio após perturbação
                - Estados crescem indefinidamente
                
                **Exemplo matemático:**
                $\\lambda_1 = 0.5 + j$, $\\lambda_2 = 0.5 - j$
                
                **Modos naturais:** $e^{0.5t}(\\cos t + j\\sin t)$
                """)
            
            st.divider()
            
            # EXEMPLO 1: SISTEMA ESTÁVEL
            st.markdown("### Exemplo 1: Sistema Estável")
            
            st.markdown("""
            **Sistema estável de 2ª ordem:**
            
            $$
            A = \\begin{bmatrix} -0.5 & 2 \\\\ -1 & -1 \\end{bmatrix}
            $$
            
            **Características:**
            - Traço: $tr(A) = -1.5$ (negativo)
            - Determinante: $\\det(A) = 2.5$ (positivo)
            - Autovalores: $\\lambda_{1,2} = -0.75 \\pm 1.39j$
            """)
            
            # Configurar sistema estável
            A_estavel = np.array([[-0.5, 2], [-1, -1]])
            autovalores_estavel = np.linalg.eigvals(A_estavel)
            
            col_estavel1, col_estavel2 = st.columns(2)
            
            with col_estavel1:
                st.markdown("**Matriz A:**")
                st.latex(f"A = \\begin{{bmatrix}} -0.5 & 2 \\\\ -1 & -1 \\end{{bmatrix}}")
                
                st.markdown("**Autovalores:**")
                for i, lam in enumerate(autovalores_estavel):
                    st.latex(f"\\lambda_{i+1} = {lam.real:.3f} {'+' if lam.imag >= 0 else ''}{lam.imag:.3f}j")
                
                st.success("✅ **SISTEMA ESTÁVEL**")
                st.caption("Ambos os autovalores têm parte real negativa")
            
            with col_estavel2:
                # Simulação com tempo longo para mostrar convergência
                def sistema_estavel(t, x):
                    return A_estavel.dot(x)
                
                # Condições iniciais
                x0_estavel = [1.0, 0.5]
                t_span_estavel = (0, 20)  # Tempo longo para mostrar convergência completa
                t_eval_estavel = np.linspace(0, 20, 1000)
                
                sol_estavel = solve_ivp(sistema_estavel, t_span_estavel, x0_estavel,
                                    method='RK45', t_eval=t_eval_estavel,
                                    max_step=0.1, atol=1e-8, rtol=1e-8)
                
                fig_estavel, ax_estavel = plt.subplots(figsize=(8, 4))
                
                ax_estavel.plot(sol_estavel.t, sol_estavel.y[0], 'b-', linewidth=2, label='$x_1(t)$')
                ax_estavel.plot(sol_estavel.t, sol_estavel.y[1], 'r-', linewidth=2, label='$x_2(t)$')
                ax_estavel.axhline(0, color='k', alpha=0.3, linestyle='--', linewidth=0.8)
                
                # Linha de convergência exponencial
                taxa_dec = abs(autovalores_estavel[0].real)
                envelope = np.max(np.abs(x0_estavel)) * np.exp(-taxa_dec * sol_estavel.t)
                ax_estavel.plot(sol_estavel.t, envelope, 'g--', alpha=0.5, linewidth=1.5, 
                            label=f'Envelope $e^{{-{taxa_dec:.2f}t}}$')
                ax_estavel.plot(sol_estavel.t, -envelope, 'g--', alpha=0.5, linewidth=1.5)
                
                ax_estavel.set_xlabel('Tempo (s)', fontsize=11)
                ax_estavel.set_ylabel('Valor dos Estados', fontsize=11)
                ax_estavel.set_title('Sistema Estável: Convergência para Zero', fontsize=12)
                ax_estavel.legend(fontsize=10)
                ax_estavel.grid(True, alpha=0.2)
                ax_estavel.set_xlim([0, 20])
                
                st.pyplot(fig_estavel)
            
            # Análise da convergência
            st.markdown("#### Análise da Convergência")
            
            taxa_convergencia = abs(autovalores_estavel[0].real)
            tempo_acomodacao = 4 / taxa_convergencia
            
            col_conv1, col_conv2, col_conv3 = st.columns(3)
            with col_conv1:
                st.metric("Taxa de convergência", f"{taxa_convergencia:.3f} s⁻¹")
            with col_conv2:
                st.metric("Tempo de acomodação (2%)", f"{tempo_acomodacao:.1f} s")
            with col_conv3:
                freq_oscilacao = abs(autovalores_estavel[0].imag) / (2 * np.pi)
                st.metric("Frequência de oscilação", f"{freq_oscilacao:.2f} Hz")
            
            st.markdown("""
            **Observação:** 
            O sistema converge exponencialmente para zero. Note como ambas as variáveis de estado 
            permanecem dentro do envelope exponencial $e^{-0.75t}$, demonstrando a estabilidade assintótica.
            """)
            
            st.divider()
            
            # EXEMPLO 2: SISTEMA INSTÁVEL
            st.markdown("### Exemplo 2: Sistema Instável")
            
            st.markdown("""
            **Sistema instável de 2ª ordem:**
            
            $$
            A = \\begin{bmatrix} 0.5 & 1 \\\\ -1 & 0.5 \\end{bmatrix}
            $$
            
            **Características:**
            - Traço: $tr(A) = 1.0$ (positivo)
            - Determinante: $\\det(A) = 1.25$ (positivo)
            - Autovalores: $\\lambda_{1,2} = 0.5 \\pm j$
            """)
            
            # Configurar sistema instável
            A_instavel = np.array([[0.5, 1], [-1, 0.5]])
            autovalores_instavel = np.linalg.eigvals(A_instavel)
            
            col_instavel1, col_instavel2 = st.columns(2)
            
            with col_instavel1:
                st.markdown("**Matriz A:**")
                st.latex(f"A = \\begin{{bmatrix}} 0.5 & 1 \\\\ -1 & 0.5 \\end{{bmatrix}}")
                
                st.markdown("**Autovalores:**")
                for i, lam in enumerate(autovalores_instavel):
                    st.latex(f"\\lambda_{i+1} = {lam.real:.3f} {'+' if lam.imag >= 0 else ''}{lam.imag:.3f}j")
                
                st.error("⚠️ **SISTEMA INSTÁVEL**")
                st.caption("Autovalores com parte real positiva")
            
            with col_instavel2:
                # Simulação com tempo moderado (divergência rápida)
                def sistema_instavel(t, x):
                    return A_instavel.dot(x)
                
                # Condições iniciais pequenas
                x0_instavel = [0.1, 0.05]  # Valores pequenos para visualização
                t_span_instavel = (0, 10)  # Tempo mais curto devido à divergência
                t_eval_instavel = np.linspace(0, 10, 1000)
                
                sol_instavel = solve_ivp(sistema_instavel, t_span_instavel, x0_instavel,
                                        method='RK45', t_eval=t_eval_instavel,
                                        max_step=0.05, atol=1e-8, rtol=1e-8)
                
                fig_instavel, ax_instavel = plt.subplots(figsize=(8, 4))
                
                ax_instavel.plot(sol_instavel.t, sol_instavel.y[0], 'b-', linewidth=2, label='$x_1(t)$')
                ax_instavel.plot(sol_instavel.t, sol_instavel.y[1], 'r-', linewidth=2, label='$x_2(t)$')
                
                # Linha de divergência exponencial
                taxa_cresc = autovalores_instavel[0].real
                envelope_cresc = np.max(np.abs(x0_instavel)) * np.exp(taxa_cresc * sol_instavel.t)
                ax_instavel.plot(sol_instavel.t, envelope_cresc, 'g--', alpha=0.5, linewidth=1.5,
                                label=f'Envelope $e^{{{taxa_cresc:.2f}t}}$')
                ax_instavel.plot(sol_instavel.t, -envelope_cresc, 'g--', alpha=0.5, linewidth=1.5)
                
                ax_instavel.set_xlabel('Tempo (s)', fontsize=11)
                ax_instavel.set_ylabel('Valor dos Estados', fontsize=11)
                ax_instavel.set_title('Sistema Instável: Divergência Exponencial', fontsize=12)
                ax_instavel.legend(fontsize=10)
                ax_instavel.grid(True, alpha=0.2)
                ax_instavel.set_xlim([0, 10])
                
                # Ajustar limites do eixo y para acomodar crescimento
                y_max = np.max(np.abs(sol_instavel.y)) * 1.1
                ax_instavel.set_ylim([-y_max, y_max])
                
                st.pyplot(fig_instavel)
            
            # Análise da divergência
            st.markdown("#### Análise da Divergência")
            
            taxa_divergencia = autovalores_instavel[0].real
            tempo_dobro = np.log(2) / taxa_divergencia
            
            col_div1, col_div2, col_div3 = st.columns(3)
            with col_div1:
                st.metric("Taxa de crescimento", f"{taxa_divergencia:.3f} s⁻¹")
            with col_div2:
                st.metric("Tempo para dobrar", f"{tempo_dobro:.1f} s")
            with col_div3:
                freq_oscilacao_inst = abs(autovalores_instavel[0].imag) / (2 * np.pi)
                st.metric("Frequência de oscilação", f"{freq_oscilacao_inst:.2f} Hz")
            
            st.markdown("""
            **Observação:** 
            O sistema diverge exponencialmente. Mesmo partindo de condições iniciais pequenas 
            ($x_1(0)=0.1$, $x_2(0)=0.05$), os estados crescem rapidamente. O tempo para dobrar 
            de valor é de aproximadamente ${tempo_dobro:.1f}$ segundos.
            """)
            
            st.divider()
            
            # EXEMPLO 3: TRANSFORMAÇÃO DE ESTABILIDADE
            st.markdown("### Exemplo 3: Sistema com Transição de Estabilidade")
            
            st.markdown("""
            **Sistema com parâmetro variável $k$:**
            
            $$
            A(k) = \\begin{bmatrix} -1 & 2 \\\\ -k & -1 \\end{bmatrix}
            $$
            
            **Análise:**
            - Determinante: $\\det(A) = 1 - 2k$
            - Autovalores: $\\lambda_{1,2} = -1 \\pm \\sqrt{2k-1}$ para $k \\geq 0.5$
            """)
            
            k_var = st.slider("Parâmetro k", -1.0, 2.0, 0.5, 0.1,
                            help="Varie k para observar a transição de estabilidade",
                            key="k_transicao")
            
            # Calcular sistema para k atual
            A_var = np.array([[-1, 2], [-k_var, -1]])
            autovalores_var = np.linalg.eigvals(A_var)
            det_var = 1 - 2*k_var
            
            col_trans1, col_trans2 = st.columns(2)
            
            with col_trans1:
                st.markdown("**Sistema atual:**")
                st.latex(f"A({k_var:.1f}) = \\begin{{bmatrix}} -1 & 2 \\\\ {-k_var:.1f} & -1 \\end{{bmatrix}}")
                
                st.markdown("**Propriedades:**")
                st.markdown(f"- Determinante: $\\det(A) = {det_var:.2f}$")
                st.markdown(f"- Autovalores:")
                for i, lam in enumerate(autovalores_var):
                    st.latex(f"\\quad \\lambda_{i+1} = {lam.real:.3f} {'+' if lam.imag >= 0 else ''}{lam.imag:.3f}j")
                
                # Determinar estabilidade
                re1, re2 = autovalores_var[0].real, autovalores_var[1].real
                
                if re1 < 0 and re2 < 0:
                    estado_var = "ESTÁVEL"
                    cor_var = "green"
                    explicacao = "k < 0.5: ambos autovalores negativos"
                elif abs(re1) < 0.01 and abs(re2) < 0.01:
                    estado_var = "MARGINAL"
                    cor_var = "orange"
                    explicacao = "k = 0.5: autovalores imaginários puros"
                elif re1 > 0 or re2 > 0:
                    estado_var = "INSTÁVEL"
                    cor_var = "red"
                    explicacao = "k > 0.5: um autovalor positivo"
                
                st.markdown(f"**Status:** <span style='color:{cor_var}'>{estado_var}</span>", 
                        unsafe_allow_html=True)
                st.caption(explicacao)
            
            with col_trans2:
                # Simulação para valor atual de k
                def sistema_variavel(t, x):
                    return A_var.dot(x)
                
                x0_var = [1.0, 0.0]
                
                # Ajustar tempo de simulação baseado na estabilidade
                if re1 < 0 and re2 < 0:
                    t_max_var = 15  # Tempo longo para convergência
                elif re1 > 0 or re2 > 0:
                    t_max_var = 8   # Tempo curto para divergência
                else:
                    t_max_var = 10  # Tempo médio para oscilação
                
                t_span_var = (0, t_max_var)
                t_eval_var = np.linspace(0, t_max_var, 1000)
                
                sol_var = solve_ivp(sistema_variavel, t_span_var, x0_var,
                                method='RK45', t_eval=t_eval_var,
                                max_step=0.05, atol=1e-8, rtol=1e-8)
                
                fig_var, ax_var = plt.subplots(figsize=(8, 4))
                
                ax_var.plot(sol_var.t, sol_var.y[0], 'b-', linewidth=2, label='$x_1(t)$')
                ax_var.plot(sol_var.t, sol_var.y[1], 'r-', linewidth=2, label='$x_2(t)$')
                ax_var.axhline(0, color='k', alpha=0.3, linestyle='--', linewidth=0.8)
                
                ax_var.set_xlabel('Tempo (s)', fontsize=11)
                ax_var.set_ylabel('Valor dos Estados', fontsize=11)
                ax_var.set_title(f'Resposta do Sistema para k = {k_var:.1f}', fontsize=12)
                ax_var.legend(fontsize=10)
                ax_var.grid(True, alpha=0.2)
                
                # Ajustar limites do eixo y
                y_max_var = np.max(np.abs(sol_var.y)) * 1.2
                ax_var.set_ylim([-y_max_var, y_max_var])
                
                st.pyplot(fig_var)
            
            # Tabela de transição
            st.markdown("#### Pontos Críticos da Transição")
            
            col_tab1, col_tab2, col_tab3 = st.columns(3)
            
            with col_tab1:
                st.markdown("**k < 0.5**")
                st.markdown("- Determinante > 0")
                st.markdown("- Autovalores complexos")
                st.markdown("- Parte real negativa")
                st.markdown("✅ **Estável oscilatório**")
            
            with col_tab2:
                st.markdown("**k = 0.5**")
                st.markdown("- Determinante = 0")
                st.markdown("- Autovalores imaginários")
                st.markdown("- Parte real zero")
                st.markdown("⚠️ **Marginalmente estável**")
            
            with col_tab3:
                st.markdown("**k > 0.5**")
                st.markdown("- Determinante < 0")
                st.markdown("- Autovalores reais")
                st.markdown("- Um positivo, um negativo")
                st.markdown("❌ **Instável**")
            
            st.divider()
            
            # RESUMO FINAL
            st.markdown("### Resumo e Conclusões")
            
            st.markdown("""
            **Principais Aprendizados:**
            
            1. **Autovalores determinam estabilidade:**
            - Parte real negativa → estabilidade
            - Parte real positiva → instabilidade
            - Parte real zero → estabilidade marginal
            
            2. **Comportamento temporal:**
            - Sistemas estáveis convergem exponencialmente
            - Sistemas instáveis divergem exponencialmente
            - Sistemas marginais mantêm oscilações
            
            3. **Interpretação física:**
            - Autovalores complexos indicam oscilações
            - Magnitude da parte real indica velocidade de convergência/divergência
            - Parte imaginária indica frequência de oscilação
            
            4. **Transições de estabilidade:**
            - Pequenas variações paramétricas podem mudar a estabilidade
            - Pontos críticos ocorrem quando autovalores cruzam o eixo imaginário
            - Sistemas reais devem ter margem de estabilidade
            """)
            
            st.markdown("""
            **Aplicação Prática:**
            
            No projeto de sistemas de controle, buscam-se:
            - Autovalores bem localizados no SPE (afastados do eixo imaginário)
            - Margens de estabilidade adequadas
            - Resposta temporal com características desejadas (tempo de acomodação, overshoot)
            
            A análise por autovalores é fundamental para garantir que sistemas dinâmicos 
            operem de forma segura e previsível.
            """)

        with tab_resp:
       
       
            st.subheader("Análise da Resposta Temporal - Funções de Transferência")
            
            st.markdown("""
            ### Sistemas Típicos em Controle
            
            A tabela abaixo resume as principais características dos sistemas mais comuns em engenharia de controle:
            
            | Tipo de Sistema | Função de Transferência | Características Principais |
            |----------------|------------------------|-----------------|
            | **1ª Ordem** | $G(s) = \\frac{K}{\\tau s + 1}$ | Resposta exponencial, sem oscilações |
            | **2ª Ordem** | $G(s) = \\frac{K\\omega_n^2}{s^2 + 2\\zeta\\omega_n s + \\omega_n^2}$ | Pode apresentar oscilações e overshoot |
            | **Com Tempo Morto** | $G(s) = \\frac{K e^{-\\theta s}}{\\tau s + 1}$ | Atraso puro no início da resposta |
            | **Resposta Inversa** | $G(s) = \\frac{K(1 - \\tau_a s)}{(\\tau_1 s + 1)(\\tau_2 s + 1)}$ | Direção inicial oposta à final |
            """)
            
            # Link para vídeo educativo
            st.markdown("""
            **Recurso Educativo Complementar:**
            
            Para uma explicação prática sobre análise de sistemas dinâmicos, assista ao vídeo:
            [Produto Tecnológico de Matheus Marinho: Análise de Resposta Temporal](https://youtu.be/bqn2M7J8Tsk?si=-rdwauJVCcsXINj7)
            
            *Legendas disponíveis em português.*
            """)
            
            st.divider()
            
            # SELETOR DE TIPO DE SISTEMA
            tipo_sistema = st.selectbox(
                "Selecione o tipo de sistema para análise detalhada:",
                ["Sistema de 1ª Ordem", 
                "Sistema de 2ª Ordem", 
                "Sistema com Tempo Morto", 
                "Sistema com Resposta Inversa"],
                key="tipo_sistema_resposta"
            )
            
            if tipo_sistema == "Sistema de 1ª Ordem":
                st.markdown("## Sistema de 1ª Ordem")
                
                col_1ord1, col_1ord2 = st.columns(2)
                
                with col_1ord1:
                    st.markdown("### Função de Transferência")
                    st.latex(r"G(s) = \frac{K}{\tau s + 1}")
                    
                    st.markdown("### Parâmetros do Sistema")
                    K_1ord = st.slider("Ganho K", 0.1, 5.0, 1.0, 0.1, key="K_1ord")
                    tau_1ord = st.slider("Constante de tempo τ (s)", 0.1, 10.0, 1.0, 0.1, key="tau_1ord")
                    
                    st.markdown("### Propriedades do Sistema")
                    st.markdown(f"- **Polo:** $s = -1/\\tau = {-1/tau_1ord:.2f}$")
                    st.markdown(f"- **Tempo de acomodação (2%):** $4\\tau = {4*tau_1ord:.1f}$ s")
                    st.markdown(f"- **Tempo de subida (10-90%):** $2.2\\tau = {2.2*tau_1ord:.1f}$ s")
                    st.markdown(f"- **Valor de regime permanente:** ${K_1ord:.1f}$")
                    
                    st.markdown("""
                    ### Interpretação Física
                    - **K:** Amplificação do sinal de entrada
                    - **τ:** Velocidade de resposta (quanto maior, mais lento)
                    - **63.2% do valor final:** Ocorre em $t = \\tau$
                    """)
                
                with col_1ord2:
                    # Simulação com tempo adequado
                    t_max = max(10, 5*tau_1ord)  # Garante tempo suficiente para visualização
                    t = np.linspace(0, t_max, 1000)
                    
                    # Resposta ao degrau
                    y = K_1ord * (1 - np.exp(-t/tau_1ord))
                    
                    fig_1ord, ax_1ord = plt.subplots(figsize=(9, 6))
                    
                    # Plot principal
                    ax_1ord.plot(t, y, 'b-', linewidth=3, label='Resposta do sistema')
                    ax_1ord.axhline(K_1ord, color='r', linestyle='--', alpha=0.7, 
                                linewidth=2, label=f'Valor final (K={K_1ord})')
                    
                    # Marcar 63.2%
                    y_63 = 0.632 * K_1ord
                    idx_63 = np.argmax(y >= y_63)
                    if idx_63 > 0:
                        ax_1ord.axvline(t[idx_63], color='g', linestyle=':', alpha=0.8, 
                                    linewidth=2, label=f'Constante de tempo τ = {t[idx_63]:.2f} s')
                        ax_1ord.axhline(y_63, color='g', linestyle=':', alpha=0.8, linewidth=1.5)
                        ax_1ord.plot(t[idx_63], y_63, 'go', markersize=10)
                    
                    # Marcar tempo de acomodação
                    t_acomod = 4 * tau_1ord
                    ax_1ord.axvline(t_acomod, color='orange', linestyle='--', alpha=0.6, 
                                linewidth=1.5, label=f'Tempo acomodação (2%) = {t_acomod:.1f} s')
                    
                    # Configurações do gráfico
                    ax_1ord.set_xlabel('Tempo (s)', fontsize=12)
                    ax_1ord.set_ylabel('Resposta y(t)', fontsize=12)
                    ax_1ord.set_title(f'Resposta ao Degrau - Sistema de 1ª Ordem\nK={K_1ord}, τ={tau_1ord} s', 
                                    fontsize=14, fontweight='bold')
                    ax_1ord.legend(loc='lower right', fontsize=10)
                    ax_1ord.grid(True, alpha=0.3)
                    ax_1ord.set_xlim([0, t_max])
                    ax_1ord.set_ylim([-0.1*K_1ord, 1.3*K_1ord])
                    
                    # Adicionar informações no gráfico
                    ax_1ord.text(0.02, 0.95, f'τ = {tau_1ord:.2f} s', transform=ax_1ord.transAxes,
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    st.pyplot(fig_1ord)
                    
                    # Explicação adicional
                    st.markdown("""
                    **Observações:**
                    - A resposta atinge 63.2% do valor final em $t = \\tau$
                    - 95% do valor final é alcançado em $t = 3\\tau$
                    - O sistema está sempre estável para $\\tau > 0$
                    """)
            
            elif tipo_sistema == "Sistema de 2ª Ordem":
                st.markdown("## Sistema de 2ª Ordem")
                
                col_2ord1, col_2ord2, col_2ord3 = st.columns(3)
                
                with col_2ord1:
                    K_2ord = st.slider("Ganho K", 0.1, 5.0, 1.0, 0.1, key="K_2ord")
                
                with col_2ord2:
                    zeta = st.slider("Fator de amortecimento ζ", -0.5, 2.0, 0.7, 0.1, 
                                help="ζ < 0: instável, 0 < ζ < 1: subamortecido, ζ ≥ 1: superamortecido",
                                key="zeta_2ord")
                
                with col_2ord3:
                    wn = st.slider("Frequência natural ωₙ (rad/s)", 0.5, 5.0, 2.0, 0.1, key="wn_2ord")
                
                st.markdown("### Função de Transferência")
                st.latex(f"G(s) = \\frac{{{K_2ord:.1f} \\times {wn:.1f}^2}}{{s^2 + 2\\times{zeta:.1f}\\times{wn:.1f}s + {wn:.1f}^2}}")
                
                # Análise das características
                if zeta < 0:
                    status = "**Sistema Instável Oscilatório**"
                    cor = 'red'
                    wd = wn * np.sqrt(1 - zeta**2) if abs(zeta) < 1 else 0
                    overshoot = 0
                    ts = "∞ (sistema diverge)"
                elif abs(zeta) < 0.01:
                    status = "**Sistema Não Amortecido**"
                    cor = 'orange'
                    wd = wn
                    overshoot = "∞ (oscilação sustentada)"
                    ts = "∞"
                elif 0 < zeta < 1:
                    status = "**Sistema Subamortecido**"
                    cor = 'blue'
                    wd = wn * np.sqrt(1 - zeta**2)
                    overshoot_val = 100*np.exp(-zeta*np.pi/np.sqrt(1-zeta**2))
                    overshoot = f"{overshoot_val:.1f}%"
                    ts = f"{4/(zeta*wn):.2f} s (2%)"
                else:  # zeta ≥ 1
                    status = "**Sistema Superamortecido**"
                    cor = 'green'
                    wd = 0
                    overshoot = "0%"
                    ts = f"{4/(zeta*wn):.2f} s"
                
                st.markdown(f"### Classificação: <span style='color:{cor}'>{status}</span>", unsafe_allow_html=True)
                
                col_prop1, col_prop2 = st.columns(2)
                with col_prop1:
                    st.markdown("#### Parâmetros Calculados")
                    if wd > 0:
                        st.metric("Frequência amortecida ωd", f"{wd:.2f} rad/s")
                    st.metric("Período de oscilação", f"{2*np.pi/wd:.2f} s" if wd > 0 else "N/A")
                    st.metric("Overshoot (Mp)", overshoot)
                
                with col_prop2:
                    st.markdown("#### Especificações Temporais")
                    st.metric("Tempo de subida Tr", f"{np.pi/wn:.2f} s" if zeta < 1 else "N/A")
                    st.metric("Tempo de pico Tp", f"{np.pi/wd:.2f} s" if 0 < zeta < 1 else "N/A")
                    st.metric("Tempo de acomodação Ts", ts)
                
                # Simulação
                t_max = 20 if (zeta < 1 and zeta > 0) else 10
                t = np.linspace(0, t_max, 2000)
                
                if zeta < 0:
                    # Instável oscilatório
                    y = K_2ord * (1 + np.exp(-zeta*wn*t) * np.sin(wn*t))
                elif abs(zeta) < 0.01:
                    # Não amortecido
                    y = K_2ord * (1 - np.cos(wn*t))
                elif zeta < 1:
                    # Subamortecido
                    phi = np.arccos(zeta)
                    y = K_2ord * (1 - (np.exp(-zeta*wn*t)/np.sqrt(1-zeta**2)) * np.sin(wd*t + phi))
                elif abs(zeta - 1) < 0.01:
                    # Criticamente amortecido
                    y = K_2ord * (1 - (1 + wn*t) * np.exp(-wn*t))
                else:
                    # Superamortecido
                    r1 = -zeta*wn + wn*np.sqrt(zeta**2 - 1)
                    r2 = -zeta*wn - wn*np.sqrt(zeta**2 - 1)
                    y = K_2ord * (1 - (r2*np.exp(r1*t) - r1*np.exp(r2*t))/(r2 - r1))
                
                # Gráfico
                fig_2ord, ax_2ord = plt.subplots(figsize=(11, 7))
                ax_2ord.plot(t, y, color=cor, linewidth=3)
                ax_2ord.axhline(K_2ord, color='gray', linestyle='--', alpha=0.6, 
                            linewidth=2, label=f'Valor final (K={K_2ord})')
                
                # Marcar overshoot se existir
                if 0 < zeta < 1:
                    y_max = np.max(y)
                    idx_max = np.argmax(y)
                    if y_max > K_2ord:
                        ax_2ord.plot(t[idx_max], y_max, 'ro', markersize=10)
                        ax_2ord.annotate(f'Overshoot = {overshoot}', 
                                    xy=(t[idx_max], y_max),
                                    xytext=(t[idx_max]+0.5, y_max+0.2),
                                    fontsize=11,
                                    arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5))
                
                # Marcar tempo de acomodação
                if zeta > 0:
                    ts_val = 4/(zeta*wn)
                    ax_2ord.axvline(ts_val, color='orange', linestyle='--', alpha=0.7,
                                linewidth=2, label=f'Tempo acomodação = {ts_val:.2f} s')
                    ax_2ord.axhline(K_2ord*0.98, color='orange', linestyle=':', alpha=0.5, linewidth=1)
                    ax_2ord.axhline(K_2ord*1.02, color='orange', linestyle=':', alpha=0.5, linewidth=1)
                
                ax_2ord.set_xlabel('Tempo (s)', fontsize=13)
                ax_2ord.set_ylabel('Resposta y(t)', fontsize=13)
                ax_2ord.set_title(f'Resposta ao Degrau - Sistema de 2ª Ordem\nζ={zeta:.2f}, ωₙ={wn:.1f} rad/s, K={K_2ord}', 
                                fontsize=15, fontweight='bold')
                ax_2ord.legend(fontsize=11)
                ax_2ord.grid(True, alpha=0.3)
                ax_2ord.set_xlim([0, t_max])
                
                # Adicionar informações sobre polos
                if zeta < 1:
                    polos_info = f"Polos: {-zeta*wn:.2f} ± j{wd:.2f}"
                else:
                    polos_info = f"Polos: {-zeta*wn + wn*np.sqrt(zeta**2-1):.2f}, {-zeta*wn - wn*np.sqrt(zeta**2-1):.2f}"
                
                ax_2ord.text(0.02, 0.95, polos_info, transform=ax_2ord.transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                st.pyplot(fig_2ord)
                
                # Explicação sobre amortecimento
                st.markdown("""
                ### Interpretação do Fator de Amortecimento (ζ)
                
                - **ζ < 0**: Sistema instável - resposta diverge
                - **ζ = 0**: Não amortecido - oscilações sustentadas
                - **0 < ζ < 1**: Subamortecido - oscilações decrescentes
                - **ζ = 1**: Criticamente amortecido - resposta mais rápida sem overshoot
                - **ζ > 1**: Superamortecido - resposta lenta sem oscilações
                
                **Valores típicos em sistemas de controle:** 0.4 < ζ < 0.8
                """)
            
            elif tipo_sistema == "Sistema com Tempo Morto":
                st.markdown("## Sistema com Tempo Morto (Atraso de Transporte)")
                
                col_delay1, col_delay2 = st.columns(2)
                
                with col_delay1:
                    st.markdown("### Função de Transferência")
                    st.latex(r"G(s) = \frac{K e^{-\theta s}}{\tau s + 1}")
                    
                    st.markdown("### Parâmetros do Sistema")
                    K_delay = st.slider("Ganho K", 0.1, 5.0, 1.0, 0.1, key="K_delay")
                    tau_delay = st.slider("Constante de tempo τ (s)", 0.1, 10.0, 1.0, 0.1, key="tau_delay")
                    theta = st.slider("Tempo morto θ (s)", 0.0, 10.0, 1.0, 0.1, key="theta_delay")
                    
                    st.markdown("### Análise do Sistema")
                    razao = theta/tau_delay
                    st.metric("Razão θ/τ", f"{razao:.2f}")
                    
                    if razao > 1:
                        st.markdown("**Atenção:** Sistema difícil de controlar (θ > τ)")
                        st.markdown("""
                        **Recomendações:**
                        - Use controladores com compensação de tempo morto
                        - Considere o método Smith Predictor
                        - Reduza o tempo morto se possível
                        """)
                    else:
                        st.markdown("**Sistema controlável com técnicas convencionais**")
                    
                    st.markdown("### Propriedades")
                    st.markdown(f"- **Polo:** $s = -1/\\tau = {-1/tau_delay:.2f}$")
                    st.markdown(f"- **Tempo de resposta efetivo:** ${theta + 4*tau_delay:.1f}$ s")
                    st.markdown(f"- **Ganho crítico (aproximado):** $K_c \\approx \\frac{0.9}{K_delay} \\frac{{\\tau}}{{\\theta}} = {0.9/K_delay * tau_delay/theta:.2f}$")
                
                with col_delay2:
                    # Simulação
                    t_max = max(15, theta + 5*tau_delay)
                    t = np.linspace(0, t_max, 2000)
                    
                    # Resposta sem atraso
                    y_sem_atraso = K_delay * (1 - np.exp(-t/tau_delay))
                    
                    # Resposta com atraso
                    y = np.zeros_like(t)
                    for i, ti in enumerate(t):
                        if ti >= theta:
                            y[i] = K_delay * (1 - np.exp(-(ti-theta)/tau_delay))
                    
                    fig_delay, ax_delay = plt.subplots(figsize=(9, 6))
                    
                    # Plot das duas respostas
                    ax_delay.plot(t, y_sem_atraso, 'b--', alpha=0.6, linewidth=2, label='Sistema sem tempo morto')
                    ax_delay.plot(t, y, 'r-', linewidth=3, label='Sistema com tempo morto')
                    
                    # Marcar tempo morto
                    ax_delay.axvline(theta, color='g', linestyle=':', alpha=0.8, 
                                linewidth=2, label=f'Tempo morto θ = {theta} s')
                    
                    # Marcar constantes de tempo
                    ax_delay.axvline(theta + tau_delay, color='orange', linestyle='--', 
                                alpha=0.6, linewidth=1.5, label=f'θ + τ = {theta + tau_delay:.1f} s')
                    ax_delay.axvline(theta + 4*tau_delay, color='purple', linestyle='-.', 
                                alpha=0.6, linewidth=1.5, label=f'θ + 4τ = {theta + 4*tau_delay:.1f} s')
                    
                    # Configurações do gráfico
                    ax_delay.set_xlabel('Tempo (s)', fontsize=12)
                    ax_delay.set_ylabel('Resposta y(t)', fontsize=12)
                    ax_delay.set_title(f'Sistema com Tempo Morto\nK={K_delay}, τ={tau_delay} s, θ={theta} s', 
                                    fontsize=14, fontweight='bold')
                    ax_delay.legend(loc='lower right', fontsize=10)
                    ax_delay.grid(True, alpha=0.3)
                    ax_delay.set_xlim([0, t_max])
                    ax_delay.set_ylim([-0.1*K_delay, 1.3*K_delay])
                    
                    # Adicionar região sombreada do tempo morto
                    ax_delay.fill_between([0, theta], -0.1*K_delay, 1.3*K_delay, 
                                        alpha=0.1, color='gray', label='Período de atraso')
                    
                    st.pyplot(fig_delay)
                
                st.markdown("### Efeito do Tempo Morto no Controle")
                st.markdown("""
                **Problemas causados pelo tempo morto:**
                1. **Atraso na resposta:** O sistema só começa a responder após θ segundos
                2. **Estabilidade reduzida:** Limita o ganho máximo do controlador
                3. **Dificuldade de sintonia:** Requer métodos especiais de sintonia
                
                **Regras práticas para controle PID:**
                
                $$ K_c = \\frac{0.9}{K} \\frac{\\tau}{\\theta}, \\quad \\tau_I = 3\\theta, \\quad \\tau_D = 0.5\\theta $$
                
                **Aplicações típicas:**
                - Processos químicos com tubulações longas
                - Sistemas de transporte
                - Processos de aquecimento com inércia térmica
                """)
            
            else:  # Sistema com Resposta Inversa
                st.markdown("## Sistema com Resposta Inversa")
                
                col_inv1, col_inv2 = st.columns(2)
                
                with col_inv1:
                    st.markdown("### Função de Transferência")
                    st.latex(r"G(s) = \frac{K(1 - \tau_a s)}{(\tau_1 s + 1)(\tau_2 s + 1)}")
                    
                    st.markdown("### Parâmetros do Sistema")
                    K_inv = st.slider("Ganho K", 0.1, 5.0, 1.0, 0.1, key="K_inv")
                    tau1 = st.slider("Constante τ₁ (s)", 0.1, 10.0, 2.0, 0.1, key="tau1_inv")
                    tau2 = st.slider("Constante τ₂ (s)", 0.1, 10.0, 1.0, 0.1, key="tau2_inv")
                    tau_a = st.slider("Constante τₐ (s)", 0.0, 5.0, 1.0, 0.1, key="taua_inv",
                                    help="Controla a intensidade da resposta inversa")
                    
                    st.markdown("### Análise do Sistema")
                    
                    if tau_a > 0:
                        st.markdown(f"**Sistema com resposta inversa** (τₐ = {tau_a:.1f} s)")
                        razao = tau_a/tau1
                        st.metric("Razão τₐ/τ₁", f"{razao:.2f}")
                        
                        if razao > 1:
                            st.markdown("**Atenção:** Resposta inversa pronunciada")
                        else:
                            st.markdown("Resposta inversa moderada")
                    else:
                        st.markdown("**Sistema normal** (sem resposta inversa)")
                    
                    st.markdown("### Propriedades")
                    st.markdown(f"- **Polos:** $s = -1/\\tau_1 = {-1/tau1:.2f}$, $s = -1/\\tau_2 = {-1/tau2:.2f}$")
                    st.markdown(f"- **Zero:** $s = 1/\\tau_a = {1/tau_a:.2f}$" if tau_a > 0 else "- **Zero:** Não possui")
                    st.markdown(f"- **Razão de constantes:** $\\tau_1/\\tau_2 = {tau1/tau2:.2f}$")
                
                with col_inv2:
                    # Simulação
                    t_max = max(15, 5*max(tau1, tau2))
                    t = np.linspace(0, t_max, 2000)
                    
                    # Resposta do sistema normal
                    if abs(tau1 - tau2) > 0.001:
                        A = 1/(tau1 - tau2)
                        y_normal = K_inv * (1 - (tau1*np.exp(-t/tau1) - tau2*np.exp(-t/tau2))/(tau1 - tau2))
                    else:
                        # Caso tau1 ≈ tau2
                        y_normal = K_inv * (1 - (1 + t/tau1) * np.exp(-t/tau1))
                    
                    # Resposta com inversa
                    if tau_a > 0:
                        if abs(tau1 - tau2) > 0.001:
                            y_inversa = y_normal - K_inv * (tau_a/(tau1 - tau2)) * (np.exp(-t/tau1) - np.exp(-t/tau2))
                        else:
                            y_inversa = y_normal - K_inv * (tau_a/tau1) * (t/tau1) * np.exp(-t/tau1)
                    else:
                        y_inversa = y_normal
                    
                    fig_inv, ax_inv = plt.subplots(figsize=(9, 6))
                    
                    # Plot das respostas
                    ax_inv.plot(t, y_normal, 'b--', alpha=0.6, linewidth=2, label='Sistema normal')
                    ax_inv.plot(t, y_inversa, 'r-', linewidth=3, label='Sistema com resposta inversa')
                    
                    # Marcar características da resposta inversa
                    if tau_a > 0:
                        # Encontrar mínimo (resposta inversa)
                        idx_min = np.argmin(y_inversa[:int(len(t)/3)])
                        if idx_min > 0 and y_inversa[idx_min] < 0:
                            ax_inv.plot(t[idx_min], y_inversa[idx_min], 'go', markersize=10)
                            ax_inv.annotate('Mínimo da resposta inversa', 
                                        xy=(t[idx_min], y_inversa[idx_min]),
                                        xytext=(t[idx_min]+1, y_inversa[idx_min]-0.2),
                                        fontsize=10,
                                        arrowprops=dict(arrowstyle='->', color='green', linewidth=1.5))
                        
                        # Encontrar cruzamento por zero
                        idx_zero = np.argmax(y_inversa > 0)
                        if idx_zero > 0:
                            ax_inv.plot(t[idx_zero], y_inversa[idx_zero], 'mo', markersize=8)
                            ax_inv.annotate('Cruzamento por zero', 
                                        xy=(t[idx_zero], y_inversa[idx_zero]),
                                        xytext=(t[idx_zero]+1, y_inversa[idx_zero]+0.2),
                                        fontsize=10,
                                        arrowprops=dict(arrowstyle='->', color='magenta', linewidth=1.5))
                    
                    ax_inv.axhline(0, color='k', alpha=0.3, linestyle='-', linewidth=0.8)
                    ax_inv.axhline(K_inv, color='gray', linestyle='--', alpha=0.6, 
                                linewidth=2, label=f'Valor final (K={K_inv})')
                    
                    ax_inv.set_xlabel('Tempo (s)', fontsize=12)
                    ax_inv.set_ylabel('Resposta y(t)', fontsize=12)
                    title = f'Sistema com Resposta Inversa\nτ₁={tau1} s, τ₂={tau2} s'
                    if tau_a > 0:
                        title += f', τₐ={tau_a} s'
                    ax_inv.set_title(title, fontsize=14, fontweight='bold')
                    ax_inv.legend(loc='lower right', fontsize=10)
                    ax_inv.grid(True, alpha=0.3)
                    ax_inv.set_xlim([0, t_max])
                    
                    st.pyplot(fig_inv)
                
                st.markdown("### Exemplo Prático: Nível em Tanque com Aquecedor")
                st.markdown("""
                **Fenômeno físico:**
                
                Ao aumentar a potência do aquecedor em um tanque:
                
                1. **Efeito imediato (τₐ):** Expansão térmica → Nível **sobe** momentaneamente
                2. **Efeito dominante (τ₁, τ₂):** Evaporação aumentada → Nível **desce** após alguns segundos
                
                **Consequências para controle:**
                
                - **Resposta inicial enganosa:** Movimento oposto ao esperado
                - **Dificuldade de controle:** Pode causar instabilidade se não considerado
                - **Estratégias de controle:**
                - Uso de ação derivativa para antecipar comportamento
                - Filtros para suavizar resposta
                - Controladores preditivos
                
                **Outros exemplos:**
                - Reatores químicos com múltiplas reações
                - Sistemas térmicos com diferentes constantes de tempo
                - Processos com efeitos competitivos
                """)
            
            st.divider()
            
            # COMPARATIVO ENTRE SISTEMAS
            st.markdown("## Comparativo entre Tipos de Sistemas")
            
            if st.button("Gerar Gráfico Comparativo", key="btn_comparativo"):
                # Criar gráfico comparativo
                t_comp = np.linspace(0, 15, 2000)
                
                # 1ª Ordem
                K1, tau1 = 1.0, 1.0
                y_1ord = K1 * (1 - np.exp(-t_comp/tau1))
                
                # 2ª Ordem (subamortecido)
                zeta2, wn2 = 0.5, 2.0
                wd2 = wn2 * np.sqrt(1 - zeta2**2)
                phi2 = np.arccos(zeta2)
                y_2ord = 1 - (np.exp(-zeta2*wn2*t_comp)/np.sqrt(1-zeta2**2)) * np.sin(wd2*t_comp + phi2)
                
                # Com tempo morto
                theta3 = 2.0
                y_delay = np.zeros_like(t_comp)
                for i, ti in enumerate(t_comp):
                    if ti >= theta3:
                        y_delay[i] = 1 - np.exp(-(ti-theta3)/tau1)
                
                # Com resposta inversa
                tau1_inv, tau2_inv, taua_inv = 3.0, 1.0, 2.0
                if abs(tau1_inv - tau2_inv) > 0.001:
                    A_inv = 1/(tau1_inv - tau2_inv)
                    y_normal_comp = 1 - (tau1_inv*np.exp(-t_comp/tau1_inv) - tau2_inv*np.exp(-t_comp/tau2_inv))/(tau1_inv - tau2_inv)
                    y_inversa_comp = y_normal_comp - (taua_inv/(tau1_inv - tau2_inv)) * (np.exp(-t_comp/tau1_inv) - np.exp(-t_comp/tau2_inv))
                else:
                    y_normal_comp = 1 - (1 + t_comp/tau1_inv) * np.exp(-t_comp/tau1_inv)
                    y_inversa_comp = y_normal_comp - (taua_inv/tau1_inv) * (t_comp/tau1_inv) * np.exp(-t_comp/tau1_inv)
                
                # Plot comparativo
                fig_comp, ax_comp = plt.subplots(figsize=(13, 8))
                
                ax_comp.plot(t_comp, y_1ord, 'b-', linewidth=3, label='1ª Ordem (K=1, τ=1s)')
                ax_comp.plot(t_comp, y_2ord, 'r-', linewidth=3, label='2ª Ordem (ζ=0.5, ωₙ=2 rad/s)')
                ax_comp.plot(t_comp, y_delay, 'g-', linewidth=3, label=f'Sistema com tempo morto (θ={theta3}s)')
                ax_comp.plot(t_comp, y_inversa_comp, 'm-', linewidth=3, label=f'Sistema com resposta inversa (τₐ={taua_inv}s)')
                
                ax_comp.axhline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
                
                ax_comp.set_xlabel('Tempo (s)', fontsize=14)
                ax_comp.set_ylabel('Resposta y(t)', fontsize=14)
                ax_comp.set_title('Comparativo: Respostas de Diferentes Sistemas ao Degrau', 
                                fontsize=16, fontweight='bold')
                ax_comp.legend(fontsize=12, loc='lower right')
                ax_comp.grid(True, alpha=0.3)
                ax_comp.set_xlim([0, 15])
                ax_comp.set_ylim([-0.5, 1.5])
                
                # Adicionar anotações
                ax_comp.text(1, 0.4, 'Resposta exponencial', fontsize=11, color='blue')
                ax_comp.text(1.5, 1.3, 'Overshoot', fontsize=11, color='red')
                ax_comp.text(3, 0.2, 'Atraso inicial', fontsize=11, color='green')
                ax_comp.text(2, -0.3, 'Resposta inversa', fontsize=11, color='magenta')
                
                st.pyplot(fig_comp)
                
                st.markdown("""
                ### Análise Comparativa
                
                **Características observadas:**
                
                1. **Sistema de 1ª Ordem:**
                - Resposta suave e monótona
                - Sem overshoot
                - Atinge 63.2% do valor final em t = τ
                
                2. **Sistema de 2ª Ordem (subamortecido):**
                - Apresenta oscilações
                - Tem overshoot
                - Tempo de resposta mais rápido que 1ª ordem
                - Frequência de oscilação definida por ωₙ e ζ
                
                3. **Sistema com Tempo Morto:**
                - Atraso puro no início da resposta
                - Mesma forma que 1ª ordem, mas deslocada
                - Problemas para controle
                
                4. **Sistema com Resposta Inversa:**
                - Direção inicial oposta à final
                - Mínimo negativo antes da subida
                - Desafio significativo para controle
                """)

        if st.button("Voltar ao Início"):
            st.session_state.node = 'inicio'
            st.rerun()
elif st.session_state.node == 'modelo_teorico':
    st.header("Aula: Modelagem por Princípios Fundamentais")
    st.success("Esta abordagem usa leis da física e química para descrever um processo.")

    st.subheader("1. As Etapas de uma Modelagem Matemática")
    st.markdown("""
    Modelar um processo é uma atividade sistemática que transforma um fenômeno físico em um conjunto de equações matemáticas. O processo geralmente segue estas etapas:
    1.  **Definir o Problema e os Objetivos:** Identificar as variáveis de entrada, saída e de estado.
    2.  **Desenhar um Diagrama:** Construir um esquema do processo para visualizar as fronteiras do sistema.
    3.  **Listar as Premissas:** Formular hipóteses simplificadoras para tornar o problema matematicamente tratável.
    4.  **Aplicar os Princípios Fundamentais:** Utilizar leis de conservação (massa, energia, momento) e equações constitutivas.
    5.  **Derivar a Equação Final:** Obter as equações diferenciais (EDOs) que representam o modelo.
    6.  **Validar o Modelo:** (Passo futuro) Comparar a resposta do modelo com dados experimentais.
    """)

    st.subheader("2. A Importância das Premissas")
    st.markdown("""
    Premissas são o alicerce do nosso modelo. Elas definem os limites de validade e a complexidade do problema. Adotar premissas é uma das habilidades mais importantes de um engenheiro.
    * **Por que usar?** Para simplificar a realidade, que é infinitamente complexa, em um conjunto de equações que podemos resolver.
    * **Exemplos Comuns:** Temperatura constante (isotérmico), mistura perfeita dentro de um tanque, gás se comporta como um gás ideal, não há perda de calor para o ambiente, etc.
    * **A Regra de Ouro:** Sempre declare suas premissas! Elas dizem a quem for usar seu modelo o que ele pode (e não pode) fazer.
    """)

    st.subheader("3. As Ferramentas Principais: Leis de Conservação")
    st.markdown("""
    A maioria dos modelos em engenharia nasce de uma lei de conservação. A forma dessa lei depende da propriedade que estamos analisando.
    
    **1. Balanço de Propriedades Escalares (Massa, Energia):**
    
    Para grandezas como **massa** e **energia**, que são o foco principal da Engenharia de Processos, a ferramenta central é o balanço geral em um volume de controle:
    """)
    st.latex(r"\text{ACÚMULO} = \text{ENTRADA} - \text{SAÍDA} + \text{GERAÇÃO} - \text{CONSUMO}")
    st.markdown("""
    * **Acúmulo:** A taxa de variação da propriedade dentro do sistema (ex: $\\frac{dM}{dt}$).
    * **Entrada/Saída:** Transporte da propriedade através das fronteiras (ex: vazões).
    * **Geração/Consumo:** Criação ou destruição da propriedade (ex: reações químicas, geração de calor).

    **2. Balanço de Propriedades Vetoriais (Momento/Força):**
    
    Para grandezas vetoriais como o **momento linear** (que se manifesta como **força**), a lei de conservação é a **2ª Lei de Newton**. Ela tem uma forma diferente, mas segue o mesmo princípio:
    """)
    st.latex(r"\sum \vec{F} = m\vec{a} = \frac{d(m\vec{v})}{dt}")
    st.markdown(r"""
    * **$\frac{d(m\vec{v})}{dt}$:** É o termo de **ACÚMULO** de momento linear.
    * **$\sum \vec{F}$:** É o termo de **GERAÇÃO LÍQUIDA** de momento (a soma de todas as forças externas aplicadas ao sistema, como mola, atrito, etc.).
    * Em muitos sistemas mecânicos (como o massa-mola), os termos de "Entrada/Saída" são nulos pois não há massa cruzando as fronteiras.
    """)
    
    with st.expander("Clique para ver a descrição de cada propriedade"):
        # Define o texto como uma string multilinhas
        texto_propriedades = """
        * **Dinâmico vs. Estático:** **Dinâmico** se o modelo contém derivadas, descrevendo a evolução temporal do sistema. **Estático** se descreve o sistema em regime permanente (derivadas nulas).
        * **Linear vs. Não-Linear:** **Linear** se as equações obedecem ao princípio da superposição. **Não-Linear** se contêm termos como potências, produtos de variáveis ou funções não-lineares.
        * **Forçado vs. Não-Forçado (Autônomo):** Um sistema é **Forçado** se possui uma ou mais entradas externas que afetam seu comportamento. É **Não-Forçado** ou **Autônomo** se não há entradas externas, e sua resposta depende apenas das condições iniciais.
        * **Invariante vs. Variante no Tempo:** **Invariante** se os parâmetros do modelo são constantes. **Variante** se os parâmetros mudam com o tempo.
        * **SISO vs. MISO, etc.:** Descreve a arquitetura de entradas/saídas. **SISO** (Single-Input, Single-Output), **MISO** (Multiple-Input, Single-Output), etc.
        * **Tempo-Contínuo vs. Tempo-Discreto:** **Contínuo** se descrito por equações diferenciais. **Discreto** se por equações de diferença.
        * **Parâmetros Concentrados vs. Distribuídos:** **Concentrados** se as propriedades são espacialmente uniformes (EDOs). **Distribuídos** se há variação espacial (EDPs).
        * **Determinístico vs. Estocástico:** **Determinístico** se as saídas são unicamente determinadas pelas entradas. **Estocástico** se o modelo inclui componentes aleatórios.
        """
        # Usa textwrap.dedent para remover a indentação comum
        st.markdown(textwrap.dedent(texto_propriedades))
        
    st.divider()

    # --- NAVEGAÇÃO DOS EXEMPLOS MOVIDA PARA A SIDEBAR ---
    st.sidebar.title("Estudos de Caso")
    st.sidebar.markdown("Explore os exemplos práticos por categoria:")
    tipo_sistema = st.sidebar.radio(
        "Selecione o tipo de sistema:",
        ("Sistemas de Processos", "Sistemas Elétricos", "Sistemas Mecânicos", "Sistemas Eletromecânicos"),
        key='tipo_sistema_selector'
    )
    
    # O botão de voltar principal da página
    if st.sidebar.button("⬅️ Voltar para 'Criar Modelo'", key='voltar_teorico_sidebar'):
        st.session_state.node = 'criar_modelo'
        st.rerun()

    # --- O CONTEÚDO PRINCIPAL MUDA COM BASE NA SELEÇÃO DA SIDEBAR ---
    
    if tipo_sistema == "Sistemas de Processos":
        st.header("Sistemas de Processos (Fluidos)")
        st.info(r"""
        **Caixa de Ferramentas para Sistemas de Processos:**
        A ferramenta principal aqui é o **Balanço Geral** aplicado a um Volume de Controle:
        $\text{ACÚMULO} = \text{ENTRADA} - \text{SAÍDA} + \text{GERAÇÃO} - \text{CONSUMO}$

        **1. Equações Governantes:**
        * **Balanço de massa:** $\frac{dm}{dt} = \frac{d(\rho V)}{dt} = \sum_{i=1}^{nu} \rho_i q_i - \sum_{j=1}^{ny} \rho_j q_j$
        * **Balanço molar por componente $k$:** $\frac{d(c_k V)}{dt} = \sum_{i=1}^{nu} c_{k_i} q_i - \sum_{j=1}^{ny} c_{k_j} q_j \pm \sum_{l=1}^{n} \nu_{k,l} \Gamma_l V$
        * **Balanço de momento na direção $k$:** $\frac{d(m v_k)}{dt} = \sum_{i=1}^{N} F_{i_k}$
        * **Balanço de energia:** $\frac{d(U + E_c + E_p)}{dt} = \sum_{i=1}^{nu} \rho_i q_i (h_i + E_{c_i} + E_{p_i}) - \sum_{j=1}^{ny} \rho_j q_j (h_j + E_{c_j} + E_{p_j}) \pm Q \pm W_s$

        **2. Leis Constitutivas:**
        * Lei de Fourier (Condução térmica)
        * Lei de resfriamento de Newton (Convecção térmica)
        * Lei de Arrhenius (Avanço da reação)
        """)
        # --- SEÇÃO DO TANQUE ATUALIZADA ---
        with st.expander("Exemplo 1: Tanque de Nível "):
            st.markdown("Vamos modelar a altura do líquido $h(t)$ em um tanque, assumindo que as vazões são controladas externamente.")
            
            st.subheader("1. Princípio da Conservação de Massa")
            st.markdown("Iniciamos com o balanço de massa. A taxa de acúmulo de massa no tanque é a vazão mássica de entrada ($\dot{M}_{in}$) menos a de saída ($\dot{M}_{out}$).")
            st.latex(r"\frac{dM(t)}{dt} = \dot{M}_{in}(t) - \dot{M}_{out}(t)")
            st.markdown("Onde $M(t)$ é a massa do líquido [kg] e $\dot{M}$ é a vazão mássica [kg/s].")

            st.subheader("2. Conversão para Balanço Volumétrico")
            st.markdown("Usamos a relação $M = \rho \cdot V$ (massa = densidade x volume) e $\dot{M} = \rho \cdot Q$ (vazão mássica = densidade x vazão volumétrica).")
            st.latex(r"\frac{d(\rho V(t))}{dt} = \rho Q_{in}(t) - \rho Q_{out}(t)")
            
            st.markdown("**Premissa:** Assumimos que o fluido é **incompressível**, ou seja, sua densidade ($\rho$) é constante. Assim, podemos retirá-la da derivada:")
            st.latex(r"\rho \frac{dV(t)}{dt} = \rho \left( Q_{in}(t) - Q_{out}(t) \right) \implies \frac{dV(t)}{dt} = Q_{in}(t) - Q_{out}(t)")
            st.markdown("Este é o **balanço volumétrico**.")
            
            st.subheader("3. Relação com a Altura (h)")
            st.markdown("O volume $V$ de um tanque é a área da base $A$ multiplicada pela altura $h(t)$.")
            st.latex(r"V(t) = A \cdot h(t)")
            st.markdown("Substituindo $V(t)$ no balanço volumétrico:")
            st.latex(r"\frac{d(A \cdot h(t))}{dt} = Q_{in}(t) - Q_{out}(t)")
            st.markdown("**Premissa:** Assumimos que o tanque tem **área de seção transversal (A) constante**. Assim, podemos retirá-la da derivada:")
            st.latex(r"A \frac{dh(t)}{dt} = Q_{in}(t) - Q_{out}(t)")
            
            st.subheader("4. Obtenção da EDO Final")
            st.markdown("Finalmente, isolando a derivada, temos a equação dinâmica do nível do líquido:")
            st.latex(r"\boxed{\frac{dh(t)}{dt} = \frac{Q_{in}(t) - Q_{out}(t)}{A}}")
            
            st.subheader("5. Premissas e Classificação")
            st.markdown("""
            * **Premissas (Resumo):** Fluido incompressível, área do tanque (A) constante. Ambas as vazões ($Q_{in}$ e $Q_{out}$) são bombeadas (ou seja, são entradas independentes e não dependem da altura $h$).
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pelas vazões), **MISO** (entradas $Q_{in}, Q_{out}$, saída $h$), de **Primeira Ordem** e **Invariante no Tempo**. Este é um sistema **integrador puro**.
            """)

            st.subheader("6. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown("Como as vazões de entrada e saída afetam o nível? (Assumindo que $Q_{in}$ e $Q_{out}$ são constantes).")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                Q_in_tanque = st.slider("Vazão de Entrada (Qin)", 0.0, 5.0, 2.0, 0.1, key='Qin_tanque')
            with col2:
                Q_out_tanque = st.slider("Vazão de Saída (Qout)", 0.0, 5.0, 1.5, 0.1, key='Qout_tanque')
            with col3:
                A_tanque = st.slider("Área do Tanque (A)", 0.5, 5.0, 1.0, 0.5, key='A_tanque')

            st.markdown("**Análise dos Parâmetros:**")
            st.markdown(r"""
            Este sistema é um **integrador**. Ele não tem um "valor final" estável por si só.
            * **Se $Q_{in} > Q_{out}$:** O nível **sobe linearmente** (rampa positiva). O tanque transborda.
            * **Se $Q_{in} < Q_{out}$:** O nível **desce linearmente** (rampa negativa). O tanque esvazia.
            * **Se $Q_{in} = Q_{out}$:** O nível fica **constante** (estado estacionário).
            * **Se $A \uparrow$ (tanque mais largo):** A inclinação da rampa $\left( \frac{Q_{in} - Q_{out}}{A} \right)$ diminui, tornando o processo **mais lento**.
            """)

            Q_net = Q_in_tanque - Q_out_tanque
            h0 = 10.0 # Condição inicial de altura
            t_tanque = np.linspace(0, 100, 500) # Eixo do tempo fixo
            y_tanque = (Q_net / A_tanque) * t_tanque + h0
            # Garante que o nível não fique negativo
            y_tanque = np.maximum(y_tanque, 0) 

            fig_tanque, ax_tanque = plt.subplots()
            ax_tanque.plot(t_tanque, y_tanque, label=f'Altura (h(t))')
            ax_tanque.axhline(h0, color='gray', linestyle='--', label=f'Altura Inicial (h0 = {h0} m)')
            
            if Q_net > 0:
                taxa_str = f"Enchendo ({Q_net/A_tanque:+.2f} m/s)"
            elif Q_net < 0:
                taxa_str = f"Esvaziando ({Q_net/A_tanque:+.2f} m/s)"
            else:
                taxa_str = "Estacionário (+0.00 m/s)"
            
            ax_tanque.set_title(f"Resposta do Integrador: {taxa_str}")
            ax_tanque.set_xlabel("Tempo (s)")
            ax_tanque.set_ylabel("Altura (h)")
            ax_tanque.set_ylim(0, 50) # Eixo Y Fixo
            ax_tanque.legend()
            ax_tanque.grid(True)
            st.pyplot(fig_tanque)
            plt.close(fig_tanque)

        with st.expander("Exemplo 2: Vaso de Gás Pressurizado - Modelo Não-Linear"):
                    st.markdown("Este modelo descreve a dinâmica da pressão P(t) em um vaso de volume V.")
                    st.subheader("1. Princípio da Conservação de Massa")
                    st.latex(r"\frac{dm(t)}{dt} = F_{entrada}(t) - F_{saida}(t)")
                    
                    st.subheader("2. Equações Constitutivas e Premissas")
                    st.markdown("""
                    * **Premissas:** Gás ideal, isotérmico, volume constante, parâmetros concentrados.
                    * **Parâmetros Fixos:** Volume ($V=2.0$ m³), Temperatura ($T=300$ K), Constantes de válvula ($k_1=k_2=5 \times 10^{-3}$).
                    """)
                    st.latex(r"F_{entrada} = k_1 \sqrt{P_1 - P(t)}")
                    st.latex(r"F_{saida} = k_2 \sqrt{P(t) - P_2}")
                    
                    st.subheader("3. Obtenção do Modelo Dinâmico Final")
                    st.latex(r"\boxed{\frac{dP}{dt} = \left(\frac{R T}{V \cdot MM}\right) \left( k_1 \sqrt{P_1 - P} - k_2 \sqrt{P - P_2} \right)}")
                    
                    st.subheader("4. Simulação Dinâmica Interativa (solve_ivp)")
                    st.info("Variáveis de entrada limitadas a P1 (Entrada) e P2 (Saída) para análise de perturbação.")
                    
                    # Parâmetros Fixos Internos
                    V_fixo = 2.0
                    T_fixo = 300.0
                    k1_fixo = 5e-3
                    k2_fixo = 5e-3
                    R, MM = 8.314, 0.029

                    # Controles limitados (Máximo 2 variáveis de processo)
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        p1_in = st.slider("Pressão de Entrada (P1) [Pa]", 200000.0, 1000000.0, 500000.0, key='vs_p1_lim')
                    with col_v2:
                        p2_out = st.slider("Pressão de Saída (P2) [Pa]", 50000.0, 150000.0, 101325.0, key='vs_p2_lim')

                    # Condições de Simulação
                    p0_init = st.number_input("Pressão Inicial P(0) [Pa]", value=150000.0, key='vs_p0_lim')
                    t_sim_final = st.slider("Tempo de Simulação [s]", 10, 500, 150, key='vs_time_lim')

                    # Definição da EDO
                    def gas_vessel_ode_lim(t, y, P1, P2):
                        P = y[0]
                        term_in = k1_fixo * np.sqrt(np.maximum(0, P1 - P))
                        term_out = k2_fixo * np.sqrt(np.maximum(0, P - P2))
                        dpdt = ((R * T_fixo) / (V_fixo * MM)) * (term_in - term_out)
                        return [dpdt]

                    # Resolução numérica
                    sol_vs = solve_ivp(gas_vessel_ode_lim, [0, t_sim_final], [p0_init], args=(p1_in, p2_out), t_eval=np.linspace(0, t_sim_final, 500))

                    # Gráfico
                    fig_vs, ax_vs = plt.subplots(figsize=(10, 4))
                    ax_vs.plot(sol_vs.t, sol_vs.y[0]/1000, color='orange', linewidth=2, label="Pressão Interna P(t)")
                    ax_vs.axhline(p1_in/1000, color='blue', linestyle='--', alpha=0.5, label="P1 (Fonte)")
                    ax_vs.axhline(p2_out/1000, color='green', linestyle='--', alpha=0.5, label="P2 (Descarga)")
                    ax_vs.set_xlabel("Tempo [s]")
                    ax_vs.set_ylabel("Pressão [kPa]")
                    ax_vs.legend()
                    ax_vs.grid(True, alpha=0.3)
                    st.pyplot(fig_vs)
                    plt.close(fig_vs)

                    st.subheader("5. Análise das Variáveis")
                    # Cálculo do ponto de equilíbrio considerando k1 = k2
                    p_ss = (p1_in + p2_out) / 2
                    st.markdown(f"""
                    * **P1 e P2 (Entradas):** Definem o gradiente de pressão. Como $k_1 = k_2$, o ponto de equilíbrio será a média aritmética entre as duas pressões de contorno.

                    """)

                    st.subheader("6. Análise e Classificação do Modelo")
                    st.markdown("Modelo **Dinâmico**, **Não-Linear** (raiz quadrada), **Forçado** (pelas pressões $P_1$ e $P_2$), **MISO** (2 entradas, 1 saída), de **Parâmetros Concentrados** e **Invariante no Tempo**.")
        with st.expander("Exemplo 3: Tanque com Aquecimento - Sistema MIMO Não-Linear"):
            st.markdown("Vamos modelar um tanque com aquecimento por camisa, considerando dinâmica de nível $L(t)$ e temperatura $T(t)$ simultaneamente.")
            
            st.subheader("1. Princípio da Conservação de Massa (Nível)")
            st.markdown("A taxa de acúmulo de massa no tanque é a vazão mássica de entrada ($\\dot{M}_{in}$) menos a de saída ($\\dot{M}_{out}$).")
            st.latex(r"\frac{dM(t)}{dt} = \dot{M}_{in}(t) - \dot{M}_{out}(t)")
            
            st.markdown("**Premissa:** Fluido incompressível ($\\rho$ constante). Usando $M = \\rho V$ e $\\dot{M} = \\rho q$:")
            st.latex(r"\rho \frac{dV(t)}{dt} = \rho q_{in}(t) - \rho q_{out}(t)")
            st.latex(r"\frac{dV(t)}{dt} = q_{in}(t) - q_{out}(t)")
            
            st.markdown("**Premissa:** Vazão de saída por gravidade (Torricelli), proporcional à raiz da altura:")
            st.latex(r"q_{out}(t) = k\sqrt{L(t)}")
            
            st.markdown("**Premissa:** Tanque com área constante ($A$), $V(t) = A L(t)$:")
            st.latex(r"A \frac{dL(t)}{dt} = q_{in}(t) - k\sqrt{L(t)}")
            
            st.subheader("2. Princípio da Conservação de Energia (Temperatura)")
            st.markdown("Balanço de energia: taxa de acúmulo = entrada - saída + geração:")
            st.latex(r"\frac{dE(t)}{dt} = \dot{E}_{in}(t) - \dot{E}_{out}(t) + \dot{Q}_{aquecedor}")
            
            st.markdown("Para líquido incompressível: $E = \\rho V c_p T$. Desprezando $E_c$ e $E_p$:")
            st.latex(r"\frac{d(\rho V c_p T)}{dt} = \rho q_{in} c_p T_{in} - \rho q_{out} c_p T + \dot{Q}_{aquecedor}")
            
            st.markdown("**Premissa:** $\\rho$, $c_p$, $A$ constantes. O aquecimento é por vapor saturado na camisa:")
            st.latex(r"\dot{Q}_{aquecedor} = \rho_j q_j \lambda_j")
            st.markdown("onde $\\rho_j$ é densidade do condensado, $q_j$ vazão do condensado, $\\lambda_j$ calor latente.")
            
            st.subheader("3. Sistema de Equações Diferenciais")
            st.markdown("Combinando as equações e simplificando:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Balanço de Massa (Nível):**")
                st.latex(r"\boxed{\frac{dL}{dt} = \frac{q_{in} - k\sqrt{L}}{A}}")
            with col2:
                st.markdown("**Balanço de Energia (Temperatura):**")
                st.latex(r"\boxed{\frac{dT}{dt} = \frac{\rho q_{in} c_p (T_{in} - T) + \rho_j q_j \lambda_j}{\rho A L c_p}}")
            
 
            
            st.subheader("4. Simulação Dinâmica Interativa")
            st.markdown("Explore como as entradas principais afetam a dinâmica do sistema (parâmetros fixos: $A = 2.0$ m², $k = 1.0$ m²·⁵/s, $\\rho = 1000$ kg/m³, $c_p = 4180$ J/(kg·K), $\\rho_j = 958$ kg/m³, $\\lambda_j = 2.26\\times10^6$ J/kg):")
            
            # Parâmetros fixos
            A = 2.0
            k = 0.1
            rho = 1000.0
            cp = 4180.0
            rho_j = 958.0
            lambda_j = 2.26e6
            
            # Controles para as duas entradas principais
            col1, col2 = st.columns(2)
            with col1:
                q_in = st.slider("Vazão de entrada $q_{in}$ [m³/s]", 0.01, 1.0, 0.15, 0.01)
            with col2:
                q_j = st.slider("Vazão de aquecimento $q_j$ [m³/s]", 0.0001, 0.05, 0.002, 0.0001)
            
            # Condições iniciais fixas
            L0 = 2.0
            T0 = 320.0
            T_in = 300.0
            
            # Tempo de simulação
            t_final = st.slider("Tempo de simulação [s]", 100, 2000, 1000, 100)
            
            # Sistema de EDOs
            def tank_system(t, y, q_in, q_j):
                L, T = y
                if L <= 0:
                    dL_dt = max(0, q_in / A)
                    dT_dt = 0
                else:
                    dL_dt = (q_in - k * np.sqrt(L)) / A
                    dT_dt = (rho * q_in * cp * (T_in - T) + rho_j * q_j * lambda_j) / (rho * A * L * cp)
                return [dL_dt, dT_dt]
            
            # Resolver
            t_eval = np.linspace(0, t_final, 500)
            sol = solve_ivp(
                tank_system, 
                [0, t_final], 
                [L0, T0], 
                args=(q_in, q_j),
                t_eval=t_eval,
                method='RK45'
            )
            
            # Gráficos
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Nível
            ax1.plot(sol.t, sol.y[0], 'b-', linewidth=2)
            ax1.axhline(L0, color='gray', linestyle='--', alpha=0.5, label=f'L₀ = {L0} m')
            ax1.set_ylabel('Nível L(t) [m]')
            ax1.set_title(f'Dinâmica do Nível (q_in = {q_in} m³/s)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Temperatura
            ax2.plot(sol.t, sol.y[1], 'r-', linewidth=2)
            ax2.axhline(T0, color='gray', linestyle='--', alpha=0.5, label=f'T₀ = {T0} K')
            ax2.axhline(T_in, color='g', linestyle=':', alpha=0.7, label=f'T_in = {T_in} K')
            ax2.set_xlabel('Tempo [s]')
            ax2.set_ylabel('Temperatura T(t) [K]')
            ax2.set_title(f'Dinâmica da Temperatura (q_j = {q_j} m³/s)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("**Análise da Dinâmica:**")
            st.markdown(r"""
            * **Vazão de entrada ($q_{in}$):** 
            - Controla diretamente a taxa de variação do nível
            - Alto $q_{in}$ → nível aumenta rapidamente
            - Baixo $q_{in}$ → nível diminui (se $q_{in} < k\sqrt{L}$)
            
            * **Vazão de aquecimento ($q_j$):**
            - Controla a taxa de aquecimento do tanque
            - Alto $q_j$ → temperatura aumenta rapidamente
            - Baixo $q_j$ → temperatura se aproxima de $T_{in}$
            
            * **Interação entre variáveis:**
            - O nível $L$ aparece no denominador da equação de temperatura
            - Baixo nível → maior variação de temperatura para o mesmo aquecimento
            - Sistema acoplado não-linear
            """)
            st.subheader("4. Análise e Classificação do Modelo")
            st.markdown("Modelo **Dinâmico**, **Não-Linear** (devido aos termos $\sqrt{L}$ e $L$ no denominador), **Forçado** (pelas vazões $q_{in}$, $q_j$ e temperatura $T_{in}$), **MIMO** (entradas $q_{in}$, $T_{in}$, $q_j$, saídas $L$, $T$), de **Parâmetros Concentrados** e **Invariante no Tempo**.")
                        
    elif tipo_sistema == "Sistemas Elétricos":
        st.header("Sistemas Elétricos")
        st.info(r"""
        **Caixa de Ferramentas para Sistemas Elétricos:**
        
        **1. Leis Fundamentais (Leis de Kirchhoff):**
        * **LKC (Lei dos Nós):** $\sum i_{entra} = \sum i_{sai}$ (Conservação de Carga)
        * **LKT (Lei das Malhas):** $\sum V = 0$ (Conservação de Energia)
        
        **2. Relações Constitutivas (Componentes):**
        * **Resistor (R):** $V_R = R \cdot i$ (Lei de Ohm)
        * **Capacitor (C):** $V_C = \frac{q}{C}$ ou $i = C \frac{dV_C}{dt}$
        * **Indutor (L):** $V_L = L \frac{di}{dt}$

        **3. Definição das Variáveis:**
        * **$\epsilon(t)$:** Tensão da fonte (aplicada) [V]
        * **$V(t)$:** Tensão (queda de potencial) [V]
        * **$i(t)$:** Corrente elétrica [A]
        * **$q(t)$:** Carga elétrica no capacitor [C]
        * **$R$:** Resistência [$\Omega$]
        * **$L$:** Indutância [H]
        * **$C$:** Capacitância [F]
        """)
        with st.expander("Exemplo 4: Reator Químico CSTR com Camisa - Sistema MIMO"):
            st.markdown("Vamos modelar um reator de mistura perfeita (CSTR) com uma reação exotérmica $A \rightarrow B$ e resfriamento por camisa.")
            
            st.subheader("1. Princípio da Conservação de Massa e Espécie")
            st.markdown("Considerando volume ($V$) e densidade ($\rho$) constantes (Balanço Global em regime permanente de massa):")
            st.latex(r"q_1 = q")
            st.markdown("Balanço por espécie $A$ (Acúmulo = Entrada - Saída - Reação):")
            st.latex(r"V \frac{dC_A}{dt} = q_1(C_{A,1} - C_A) - \Gamma V")
            
            st.subheader("2. Princípio da Conservação de Energia")
            st.markdown("Balanço de energia no reator e na camisa de resfriamento:")
            st.latex(r"\rho V c_p \frac{dT}{dt} = \rho q_1 c_p (T_1 - T) + (-\Delta H_r) \Gamma V + UA(T_c - T)")
            st.latex(r"\rho_c V_c c_{p,c} \frac{dT_c}{dt} = \rho_c q_c c_{p,c} (T_{c,0} - T_c) - UA(T_c - T)")

            st.subheader("3. Equações Constitutivas (Lei de Arrhenius)")
            st.markdown("A taxa de reação ($\Gamma$) depende fortemente da temperatura:")
            st.latex(r"\Gamma = k_0 \exp\left(-\frac{E}{RT}\right) C_A")

            st.subheader("4. Simulação Dinâmica Interativa (solve_ivp)")
            st.info("Explore como a vazão de alimentação (q1) e a vazão de resfriamento (qc) afetam a conversão e a temperatura.")

            # Parâmetros Fixos (Baseados em literatura de reatores)
            V = 100.0        # L
            Vc = 20.0        # L
            rho = 1000.0     # g/L
            cp = 4.18        # J/(g.K)
            k0 = 7.2e10      # 1/min
            E_R = 8750.0     # K (E/R)
            dH = -50000.0    # J/mol (Exotérmica)
            UA = 5000.0      # J/(min.K)
            CA1 = 1.0        # mol/L
            T1 = 350.0       # K
            Tc0 = 300.0      # K

            col1, col2 = st.columns(2)
            with col1:
                q1 = st.slider("Vazão de Alimentação $q_1$ [L/min]", 5.0, 50.0, 10.0, key='r_q1')
            with col2:
                qc = st.slider("Vazão de Resfriamento $q_c$ [L/min]", 5.0, 100.0, 15.0, key='r_qc')

            def cstr_system(t, y, q1, qc):
                Ca, T, Tc = y
                # Cinética
                Gamma = k0 * np.exp(-E_R / T) * Ca
                
                # EDOs
                dCadt = (q1/V)*(CA1 - Ca) - Gamma
                dTdt = (q1/V)*(T1 - T) + (-dH*Gamma)/(rho*cp) + (UA/(rho*V*cp))*(Tc - T)
                dTcdt = (qc/Vc)*(Tc0 - Tc) - (UA/(rho*Vc*cp))*(Tc - T)
                
                return [dCadt, dTdt, dTcdt]

            # Resolver
            t_final_r = st.slider("Tempo de Simulação [min]", 1, 100, 30)
            sol_r = solve_ivp(cstr_system, [0, t_final_r], [0.8, 350.0, 305.0], args=(q1, qc), t_eval=np.linspace(0, t_final_r, 500))

            # Gráficos
            fig_r, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(sol_r.t, sol_r.y[0], 'g-', label='Concentração $C_A$ [mol/L]')
            ax1.set_ylabel('Concentração')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(sol_r.t, sol_r.y[1], 'r-', label='Temp. Reator (T)')
            ax2.plot(sol_r.t, sol_r.y[2], 'b--', label='Temp. Camisa (Tc)')
            ax2.set_xlabel('Tempo [min]')
            ax2.set_ylabel('Temperatura [K]')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig_r)

            st.subheader("5. Análise das Variáveis")
            st.markdown(r"""
            * **Vazão $q_1$:** Afeta o tempo de residência ($V/q_1$). Aumentar $q_1$ reduz o tempo para a reação ocorrer, mas traz mais reagente fresco.
            * **Vazão $q_c$:** É a principal variável de manipulação para controle térmico. O aumento de $q_c$ remove o calor gerado pela reação exotérmica através do termo $UA(T_c - T)$.
            * **Acoplamento:** A concentração $C_A$ e a temperatura $T$ são fortemente acopladas através do termo de reação não-linear (Arrhenius).
            """)

            st.subheader("6. Análise e Classificação do Modelo")
            st.markdown("Modelo **Dinâmico**, **Não-Linear** (devido à exponencial de Arrhenius e produtos entre variáveis), **Forçado** (pelas vazões e temperaturas de entrada), **MIMO** (entradas $q_1, q_c$; saídas $C_A, T, T_c$), de **Parâmetros Concentrados** e **Invariante no Tempo**.")

        with st.expander("Exemplo 1: Circuito RC (Malha Única)"):
            st.markdown("Vamos modelar a carga `q(t)` no capacitor em um circuito RC série com fonte $\epsilon$.")
            st.subheader("1. Princípio Fundamental (LKT)")
            st.latex(r"V_R + V_C = \epsilon")
            st.subheader("2. Relações Constitutivas")
            st.latex(r"V_R = R \cdot i \quad \text{e} \quad V_C = \frac{q}{C}")
            st.subheader("3. Obtenção da EDO")
            st.markdown("Substituindo na LKT e usando $ i = dq/dt $:")
            st.latex(r"R \frac{dq(t)}{dt} + \frac{1}{C} q(t) = \epsilon")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Componentes (R, C) e fonte ($\epsilon$) são ideais e seus valores são constantes.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela fonte $\epsilon$), **SISO** (entrada $\epsilon$, saída $q$), de **Primeira Ordem** e **Invariante no Tempo**.
            """)
            
            st.subheader("5. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown("A constante de tempo $\tau = RC$ define a velocidade de carga do capacitor.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                R_rc = st.slider("Resistência (R)", 0.1, 5.0, 1.0, 0.1, key='R_rc')
            with col2:
                C_rc = st.slider("Capacitância (C)", 0.1, 5.0, 1.0, 0.1, key='C_rc')
            with col3:
                E_rc = st.slider("Tensão (ε)", 1.0, 10.0, 5.0, 0.5, key='E_rc')

            st.markdown("**Análise dos Parâmetros:**")
            st.markdown(r"""
            * **Carga Final ($q_\infty = C \cdot \epsilon$):** Define o valor que a carga atingirá.
                * Se $C \uparrow \implies q_\infty \uparrow$ (capacitor maior armazena mais carga).
                * Se $\epsilon \uparrow \implies q_\infty \uparrow$ (tensão maior armazena mais carga).
            * **Constante de Tempo ($\tau = R \cdot C$):** Define a "velocidade" de carga.
                * Se $R \uparrow$ (mais resistência) $\implies \tau \uparrow$ (sistema **mais lento**).
                * Se $C \uparrow$ (capacitor maior) $\implies \tau \uparrow$ (sistema **mais lento**, pois demora mais para encher).
            """)

            tau_rc = R_rc * C_rc
            q_final = C_rc * E_rc
            t_rc = np.linspace(0, 30, 500) # Eixo do tempo fixo
            y_rc = q_final * (1 - np.exp(-t_rc / tau_rc))

            fig_rc, ax_rc = plt.subplots()
            ax_rc.plot(t_rc, y_rc, label=f'Carga (q(t))')
            ax_rc.axhline(q_final, color='red', linestyle='--', label=f'Carga Final (q = {q_final:.2f} C)')
            ax_rc.axvline(tau_rc, color='gray', linestyle='--', label=f'Const. Tempo (τ = {tau_rc:.2f} s)')
            ax_rc.set_title("Carga do Capacitor (Sistema de 1ª Ordem)")
            ax_rc.set_xlabel("Tempo (s)")
            ax_rc.set_ylabel("Carga (q)")
            ax_rc.set_ylim(0, max(q_final * 1.1, 1.0))
            ax_rc.legend()
            ax_rc.grid(True)
            st.pyplot(fig_rc)
            plt.close(fig_rc)


        with st.expander("Exemplo 2: Circuito RLC (Malha Única)"):
            st.markdown("Vamos modelar a tensão no capacitor $V_C(t)$ em um circuito RLC série.")
            st.subheader("1. Princípio Fundamental (LKT)")
            st.latex(r"V_R + V_L + V_C = \epsilon")
            st.subheader("2. Relações Constitutivas")
            st.markdown("Queremos a EDO em $V_C$. Usamos as relações baseadas em $i = C \\frac{dV_C}{dt}$:")
            st.latex(r"V_R = R \cdot i = RC \frac{dV_C}{dt}")
            st.latex(r"V_L = L \frac{di}{dt} = L \frac{d}{dt}\left(C \frac{dV_C}{dt}\right) = LC \frac{d^2V_C}{dt^2}")
            
            st.subheader("3. Obtenção da EDO")
            st.markdown("Substituindo na LKT:")
            st.latex(r"\left(RC \frac{dV_C}{dt}\right) + \left(LC \frac{d^2V_C}{dt^2}\right) + V_C = \epsilon")
            st.markdown("Rearranjando para a forma canônica de 2ª ordem:")
            st.latex(r"\boxed{LC \frac{d^2V_C}{dt^2} + RC \frac{dV_C}{dt} + V_C(t) = \epsilon(t)}")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Componentes (R, L, C) e fonte ($\epsilon$) são ideais e seus valores são constantes.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela fonte $\epsilon$), **SISO** (entrada $\epsilon$, saída $V_C$), de **Segunda Ordem** e **Invariante no Tempo**.
            """)
            
            st.subheader("5. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown(r"A resposta depende de dois fatores: a **Frequência Natural ($\omega_n$)** e o **Fator de Amortecimento ($\zeta$)**.")

            col_rlc1, col_rlc2, col_rlc3 = st.columns(3)
            with col_rlc1:
                R_rlc = st.slider("Resistor (R)", 0.1, 10.0, 1.0, 0.1, key='R_rlc')
            with col_rlc2:
                L_rlc = st.slider("Indutor (L)", 0.1, 5.0, 1.0, 0.1, key='L_rlc')
            with col_rlc3:
                C_rlc = st.slider("Capacitor (C)", 0.1, 5.0, 1.0, 0.1, key='C_rlc')

            st.markdown("**Análise dos Parâmetros:**")
            st.latex(r"\omega_n = \frac{1}{\sqrt{LC}} \quad | \quad \zeta = \frac{R}{2}\sqrt{\frac{C}{L}}")
            st.markdown(r"""
            * **Se $R \uparrow$:** $\zeta \uparrow$ (sistema **mais amortecido**, menos oscilatório).
            * **Se $L \uparrow$:** $\omega_n \downarrow$ (sistema **mais lento**) e $\zeta \downarrow$ (sistema **mais oscilatório**).
            * **Se $C \uparrow$:** $\omega_n \downarrow$ (sistema **mais lento**) e $\zeta \uparrow$ (sistema **mais amortecido**).
            """)

            omega_n = 1 / np.sqrt(L_rlc * C_rlc)
            zeta = (R_rlc / 2) * np.sqrt(C_rlc / L_rlc)
            Kp = 1.0 

            t_rlc = np.linspace(0, 30, 500) # Eixo do tempo fixo
            
            if omega_n < 0.01:
                y_rlc = np.full_like(t_rlc, Kp)
                regime = "Indefinido (L ou C muito pequeno)"
            elif zeta < 1: # Subamortecido
                wd = omega_n * np.sqrt(1 - zeta**2)
                y_rlc = Kp * (1 - (np.exp(-zeta*omega_n*t_rlc) / np.sqrt(1-zeta**2)) * np.sin(wd*t_rlc + np.arccos(zeta)))
                regime = "Subamortecido (Oscilatório)"
            elif zeta == 1: # Criticamente Amortecido
                y_rlc = Kp * (1 - (1 + omega_n*t_rlc) * np.exp(-omega_n*t_rlc))
                regime = "Criticamente Amortecido (Rápido)"
            else: # Superamortecido
                p1 = -zeta*omega_n + omega_n*np.sqrt(zeta**2-1)
                p2 = -zeta*omega_n - omega_n*np.sqrt(zeta**2-1)
                if np.abs(p2 - p1) < 1e-6:
                    y_rlc = Kp * (1 - (1 + omega_n*t_rlc) * np.exp(-omega_n*t_rlc))
                    regime = "Próximo ao Crítico"
                else:
                    y_rlc = Kp * (1 + (p1*np.exp(p2*t_rlc) - p2*np.exp(p1*t_rlc))/(p2-p1))
                regime = "Superamortecido (Hiper-amortecido)"

            fig_rlc, ax_rlc = plt.subplots()
            ax_rlc.plot(t_rlc, y_rlc, label=f'Resposta (Vc(t))')
            ax_rlc.axhline(Kp, color='red', linestyle='--', label=f'Valor Final (Kp = {Kp})')
            ax_rlc.set_title(f"Resposta de 2ª Ordem: {regime}")
            ax_rlc.set_xlabel("Tempo (s)")
            ax_rlc.set_ylabel("Tensão no Capacitor (Vc)")
            ax_rlc.set_ylim(-0.5, 2.0) # Eixo Y fixo
            ax_rlc.legend()
            ax_rlc.grid(True)
            st.pyplot(fig_rlc)
            st.markdown(f"**Fator de Amortecimento Calculado ($\zeta$): {zeta:.3f}**")
            plt.close(fig_rlc)
            
        with st.expander("Exemplo 3: Circuitos RC (Malhas em Paralelo)"):
            st.markdown("Vamos modelar a tensão no capacitor $V_C(t)$ para este circuito.")
            st.subheader("1. Princípios Fundamentais (LKT e LKC)")
            st.latex(r"\text{(Malha 1): } V_C + V_{R_1} = \epsilon \implies V_{R_1} = \epsilon - V_C")
            st.latex(r"\text{(Malha 2): } V_{R_1} = V_{R_2} \implies V_{R_2} = \epsilon - V_C")
            st.latex(r"\text{(Nó): } i = i_1 + i_2")
            st.subheader("2. Relações Constitutivas")
            st.latex(r"i = C \frac{dV_C}{dt}")
            st.latex(r"i_1 = \frac{V_{R_1}}{R_1} = \frac{\epsilon - V_C}{R_1}")
            st.latex(r"i_2 = \frac{V_{R_2}}{R_2} = \frac{\epsilon - V_C}{R_2}")
            st.subheader("3. Obtenção da EDO")
            st.markdown("Substituindo na lei do nó ($i = i_1 + i_2$):")
            st.latex(r"C \frac{dV_C}{dt} = \frac{\epsilon - V_C}{R_1} + \frac{\epsilon - V_C}{R_2}")
            
            st.markdown("Definindo a resistência equivalente $R_{eq}$:")
            st.latex(r"R_{eq} = \left(\frac{1}{R_1} + \frac{1}{R_2}\right)^{-1}")
            st.markdown("Chegamos à EDO de 1ª ordem:")
            st.latex(r"\boxed{R_{eq}C \frac{dV_C}{dt} + V_C(t) = \epsilon(t)}")

            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Componentes (R₁, R₂, C) e fonte ($\epsilon$) são ideais e seus valores são constantes.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela fonte $\epsilon$), **SISO** (entrada $\epsilon$, saída $V_C$), de **Primeira Ordem** e **Invariante no Tempo**.
            """)

            st.subheader("5. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown("A constante de tempo $\tau = R_{eq}C$.")
            
            col_rcp1, col_rcp2, col_rcp3 = st.columns(3)
            with col_rcp1:
                R1_rcp = st.slider("Resistor 1 (R1)", 0.1, 5.0, 2.0, 0.1, key='R1_rcp')
            with col_rcp2:
                R2_rcp = st.slider("Resistor 2 (R2)", 0.1, 5.0, 2.0, 0.1, key='R2_rcp')
            with col_rcp3:
                C_rcp = st.slider("Capacitor (C)", 0.1, 5.0, 1.0, 0.1, key='C_rcp')

            st.markdown("**Análise dos Parâmetros:**")
            st.latex(r"R_{eq} = \left(\frac{1}{R_1} + \frac{1}{R_2}\right)^{-1} \quad | \quad \tau = R_{eq} \cdot C")
            st.markdown(r"""
            * **Se $R_1 \uparrow$ ou $R_2 \uparrow$:** A resistência equivalente $R_{eq} \uparrow$. Isso faz $\tau \uparrow$ (sistema **mais lento**).
            * **Se $C \uparrow$:** $\tau \uparrow$ (sistema **mais lento**).
            * Adicionar um resistor em paralelo (diminuir $R_2$ de infinito para um valor) **diminui** $R_{eq}$ e torna o sistema **mais rápido**.
            """)
            
            Req = 1 / (1/R1_rcp + 1/R2_rcp)
            tau_rcp = Req * C_rcp
            Kp_rcp = 1.0 

            t_rcp = np.linspace(0, 30, 500) # Eixo do tempo fixo
            y_rcp = Kp_rcp * (1 - np.exp(-t_rcp / tau_rcp))

            fig_rcp, ax_rcp = plt.subplots()
            ax_rcp.plot(t_rcp, y_rcp, label=f'Resposta (Vc(t))')
            ax_rcp.axhline(Kp_rcp, color='red', linestyle='--', label=f'Valor Final (Kp = {Kp_rcp:.2f})')
            ax_rcp.axvline(tau_rcp, color='gray', linestyle='--', label=f'Const. Tempo (τ = {tau_rcp:.2f} s)')
            ax_rcp.set_title("Carga do Capacitor (RC Paralelo)")
            ax_rcp.set_xlabel("Tempo (s)")
            ax_rcp.set_ylabel("Tensão no Capacitor (Vc)")
            ax_rcp.set_ylim(0, 1.1) # Eixo Y Fixo
            ax_rcp.legend()
            ax_rcp.grid(True)
            st.pyplot(fig_rcp)
            st.markdown(f"**Resistência Equivalente (Req): {Req:.3f} $\Omega$**")
            plt.close(fig_rcp)


    elif tipo_sistema == "Sistemas Mecânicos":
        st.header("Sistemas Mecânicos")
        st.info(r"""
        **Caixa de Ferramentas para Sistemas Mecânicos:**
        
        **1. Leis Fundamentais:**
        * **Formulação Newtoniana (Vetorial):**
            * **Translacional (2ª Lei):** $\sum F = m \cdot a = m \frac{d^2z}{dt^2}$ (Conservação de Momento Linear)
            * **Rotacional (Newton-Euller):** $\sum \tau = J \cdot \alpha = J \frac{d^2\theta}{dt^2}$ (Conservação de Momento Angular)
        * **Formulação Lagrangiana (Escalar):**
            * Define a Lagrangiana: $L = K - P$ (Energia Cinética - Energia Potencial)
            * Aplica a Equação de Lagrange: $\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = Q_{nc}$ (onde $q$ é a coordenada generalizada e $Q_{nc}$ são as forças não-conservativas).
        
        **2. Relações Constitutivas (Componentes):**
        * **Mola (k):** $F_k = k \cdot z$ | Energia Potencial: $P = \frac{1}{2}kz^2$
        * **Amortecedor (c):** $F_c = c \cdot v = c \frac{dz}{dt}$
        * **Massa (m):** Energia Cinética: $K = \frac{1}{2}mv^2$
        
        **3. Definição das Variáveis:**
        * **$F(t)$:** Força externa (aplicada) [N]
        * **$z(t), x(t)$:** Posição [m]
        * **$v(t), \dot{x}(t)$:** Velocidade [m/s]
        * **$\theta(t)$:** Ângulo [rad]
        * **$\omega(t), \dot{\theta}(t)$:** Velocidade angular [rad/s]
        * **$m, M$:** Massa [kg]
        * **$k$:** Constante da mola [N/m]
        * **$c, B$:** Coeficiente de amortecimento [N·s/m]
        * **$J$:** Momento de inércia [kg·m²]
        """)
        
        with st.expander("Exemplo 1: Sistema Massa-Mola-Amortecedor (M-C-K)"):
            st.markdown("Este é o análogo mecânico do circuito RLC. O objetivo é modelar a **resposta livre** da posição **z(t)** da massa, ou seja, sem uma força externa aplicada.")
            st.subheader("1. Princípio Fundamental (2ª Lei de Newton)")
            st.markdown("O somatório das forças é igual à massa vezes a aceleração. No caso não-forçado, as únicas forças são as de restauração da mola e do amortecedor.")
            st.latex(r"\sum F(t) = m \cdot a(t)")
            st.latex(r"- F_k(t) - F_c(t) = m \frac{d^2z}{dt^2}")
            
            st.subheader("2. Relações Constitutivas")
            st.latex(r"F_k = k \cdot z(t) \quad | \quad F_c = c \frac{dz(t)}{dt}")

            st.subheader("3. Obtenção da EDO")
            st.markdown("Substituindo as forças na 2ª Lei de Newton e rearranjando:")
            st.latex(r"- k z(t) - c \frac{dz(t)}{dt} = m \frac{d^2z(t)}{dt^2}")
            st.markdown("Na forma canônica (igual à do slide):")
            st.latex(r"\boxed{m \frac{d^2z}{dt^2} + c \frac{dz}{dt} + k z(t) = 0}")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Massa, mola e amortecedor são ideais (lineares) e seus parâmetros são constantes. O movimento é em uma única dimensão.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Não-Forçado** (autônomo), **SISO** (sem entrada, saída $z$), de **Segunda Ordem** e **Invariante no Tempo**.
            """)

            st.subheader("5. Prévia da Resposta Livre (Interativo)")
            st.markdown(r"Simulamos a resposta do sistema a uma **condição inicial** (puxando a massa para $z(0)=1$ e soltando-a, com $\dot{z}(0)=0$). A resposta depende do **fator de amortecimento ($\zeta$)**.")
            
            col_mck1, col_mck2, col_mck3 = st.columns(3)
            with col_mck1:
                m_mck = st.slider("Massa (m)", 0.1, 10.0, 5.0, 0.1, key='m_mck')
            with col_mck2:
                c_mck = st.slider("Amortecedor (c)", 0.1, 10.0, 1.0, 0.1, key='c_mck')
            with col_mck3:
                k_mck = st.slider("Mola (k)", 0.1, 5.0, 0.5, 0.1, key='k_mck')

            st.markdown("**Análise dos Parâmetros:**")
            st.latex(r"\omega_n = \sqrt{\frac{k}{m}} \quad | \quad \zeta = \frac{c}{2\sqrt{mk}}")
            st.markdown(r"""
            * **Se $c \uparrow$:** $\zeta \uparrow$ (sistema **mais amortecido**, menos oscilatório).
            * **Se $m \uparrow$:** $\omega_n \downarrow$ (sistema **mais lento**) e $\zeta \downarrow$ (sistema **mais oscilatório**).
            * **Se $k \uparrow$:** $\omega_n \uparrow$ (sistema **mais rápido**) e $\zeta \uparrow$ (sistema **mais amortecido**).
            """)

            omega_n = np.sqrt(k_mck / m_mck)
            zeta = c_mck / (2 * np.sqrt(m_mck * k_mck))
            z0 = 1.0 
            v0 = 0.0 

            t_mck = np.linspace(0, 50, 500) # Eixo do tempo fixo
            
            if omega_n < 0.01:
                y_mck = np.full_like(t_mck, z0)
                regime = "Indefinido (m ou k muito pequeno)"
            elif zeta < 1: # Subamortecido
                wd = omega_n * np.sqrt(1 - zeta**2)
                y_mck = z0 * np.exp(-zeta*omega_n*t_mck) * (np.cos(wd*t_mck) + (zeta/np.sqrt(1-zeta**2)) * np.sin(wd*t_mck))
                regime = "Subamortecido (Oscilatório)"
            elif zeta == 1: # Criticamente Amortecido
                y_mck = z0 * np.exp(-omega_n*t_mck) * (1 + omega_n*t_mck)
                regime = "Criticamente Amortecido (Rápido)"
            else: # Superamortecido
                p1 = -zeta*omega_n + omega_n*np.sqrt(zeta**2-1)
                p2 = -zeta*omega_n - omega_n*np.sqrt(zeta**2-1)
                A1 = (v0 - p2*z0) / (p1 - p2)
                A2 = (p1*z0 - v0) / (p1 - p2)
                y_mck = A1*np.exp(p1*t_mck) + A2*np.exp(p2*t_mck)
                regime = "Superamortecido (Hiper-amortecido)"

            fig_mck, ax_mck = plt.subplots()
            ax_mck.plot(t_mck, y_mck, label=f'Posição (z(t))')
            ax_mck.axhline(0, color='red', linestyle='--', label=f'Posição de Repouso (z=0)')
            ax_mck.set_title(f"Resposta Livre: {regime}")
            ax_mck.set_xlabel("Tempo (s)")
            ax_mck.set_ylabel("Posição (z)")
            ax_mck.set_ylim(-1.1, 1.1) # Eixo Y fixo
            ax_mck.legend()
            ax_mck.grid(True)
            st.pyplot(fig_mck)
            st.markdown(f"**Fator de Amortecimento Calculado ($\zeta$): {zeta:.3f}**")
            plt.close(fig_mck)

        with st.expander("Exemplo 2: Associação de Corpos Rígidos (2 Massas)"):
            st.markdown("Modelo de dois carros (massas $M_1$ e $M_2$) conectados por uma mola ($k$) e amortecedor ($c$), com uma força $u(t)$ aplicada no segundo carro.")
            st.subheader("1. Princípio Fundamental (2ª Lei de Newton)")
            st.markdown("Aplicamos a 2ª Lei de Newton para **cada massa separadamente**.")
            st.latex(r"\text{Massa 1: } M_1 \frac{d^2x_1}{dt^2} = F_c + F_k")
            st.latex(r"\text{Massa 2: } M_2 \frac{d^2x_2}{dt^2} = u(t) - F_c - F_k")
            st.subheader("2. Relações Constitutivas")
            st.markdown("As forças da mola e amortecedor dependem da **diferença** de posição e velocidade entre as massas.")
            st.latex(r"F_k = k (x_2 - x_1)")
            st.latex(r"F_c = c (\dot{x}_2 - \dot{x}_1)")
            st.subheader("3. Obtenção das EDOs (Sistema Acoplado)")
            st.latex(r"\boxed{M_1 \frac{d^2x_1}{dt^2} + c(\dot{x}_1 - \dot{x}_2) + k(x_1 - x_2) = 0}")
            st.latex(r"\boxed{M_2 \frac{d^2x_2}{dt^2} + c(\dot{x}_2 - \dot{x}_1) + k(x_2 - x_1) = u(t)}")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Massas, mola e amortecedor são ideais e constantes. Movimento sem atrito com o solo.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela força $u$), **MIMO** (entrada $u$, saídas $x_1$ e $x_2$), de **Quarta Ordem** (duas EDOs de 2ª ordem) e **Invariante no Tempo**.
            """)
            st.info("A simulação de sistemas MIMO acoplados é complexa e requer métodos numéricos para EDOs, que serão vistos em tópicos futuros.")

        with st.expander("Exemplo 3: Pêndulo Simples (Formulação Lagrangiana)"):
            st.markdown("Modelagem do ângulo $\\theta(t)$ de um pêndulo simples de comprimento $L$ e massa $m$.")
            st.subheader("1. Princípio Fundamental (Formulação Lagrangiana)")
            st.markdown("Usamos a abordagem da energia, que é mais simples para sistemas rotacionais. A coordenada generalizada é $q = \\theta$.")
            st.markdown("**Energia Cinética ($K$):**")
            st.latex(r"x_G = L\sin\theta \implies \dot{x}_G = L\dot{\theta}\cos\theta")
            st.latex(r"y_G = -L\cos\theta \implies \dot{y}_G = L\dot{\theta}\sin\theta")
            st.latex(r"K = \frac{1}{2}m(\dot{x}_G^2 + \dot{y}_G^2) = \frac{1}{2}m(L^2\dot{\theta}^2\cos^2\theta + L^2\dot{\theta}^2\sin^2\theta) = \frac{1}{2}mL^2\dot{\theta}^2")
            st.markdown("**Energia Potencial ($P$):** (Referência no pivô, $y=0$)")
            st.latex(r"P = mgy_G = -mgL\cos\theta")
            st.markdown("**Lagrangiana ($L = K - P$):**")
            st.latex(r"L = \frac{1}{2}mL^2\dot{\theta}^2 + mgL\cos\theta")

            st.subheader("2. Obtenção da EDO")
            st.markdown("Aplicamos a Equação de Lagrange (sem forças não-conservativas, $Q=0$):")
            st.latex(r"\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\theta}}\right) - \frac{\partial L}{\partial \theta} = 0")
            st.latex(r"\frac{\partial L}{\partial \dot{\theta}} = mL^2\dot{\theta} \implies \frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\theta}}\right) = mL^2\ddot{\theta}")
            st.latex(r"\frac{\partial L}{\partial \theta} = -mgL\sin\theta")
            st.markdown("Substituindo, obtemos o modelo **não-linear** do pêndulo:")
            st.latex(r"mL^2\ddot{\theta} - (-mgL\sin\theta) = 0 \implies \boxed{\ddot{\theta}(t) + \frac{g}{L}\sin\theta(t) = 0}")

            st.subheader("3. Linearização para Pequenos Ângulos")
            st.markdown(r"Para ângulos pequenos ($\theta \approx 0$), usamos a aproximação $\sin\theta \approx \theta$. Isso nos dá o modelo **linear** do oscilador harmônico:")
            st.latex(r"\boxed{\ddot{\theta}(t) + \frac{g}{L}\theta(t) = 0}")

            st.subheader("4. Premissas e Classificação (Modelo Linearizado)")
            st.markdown(r"""
            * **Premissas:** Haste de massa desprezível, pivô sem atrito, massa pontual, movimento em 2D, **pequenos ângulos ($\sin\theta \approx \theta$)**.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Não-Forçado** de **Segunda Ordem** e **Invariante no Tempo**.
            """)
            
            st.subheader("5. Prévia da Resposta Dinâmica (Modelo Linearizado)")
            st.markdown(r"Este sistema é um oscilador ideal (não amortecido, $\zeta=0$). A frequência da oscilação $\omega_n$ depende apenas de $g$ e $L$.")

            col_pen1, col_pen2 = st.columns(2)
            with col_pen1:
                L_pen = st.slider("Comprimento (L)", 0.1, 5.0, 1.0, 0.1, key='L_pen')
            with col_pen2:
                g_pen = st.slider("Gravidade (g)", 1.0, 20.0, 9.8, 0.1, key='g_pen')

            st.markdown("**Análise dos Parâmetros:**")
            st.latex(r"\omega_n = \sqrt{\frac{g}{L}}")
            st.markdown(r"""
            * **Se $L \uparrow$ (pêndulo mais longo):** $\omega_n \downarrow$ (oscilação **mais lenta**).
            * **Se $g \uparrow$ (gravidade mais forte):** $\omega_n \uparrow$ (oscilação **mais rápida**).
            """)

            omega_n_pen = np.sqrt(g_pen / L_pen)
            
            t_pen = np.linspace(0, 30, 500) # Eixo do tempo fixo
            # Simulação da resposta a uma condição inicial (theta_0 = 0.2 rad)
            theta_0 = 0.2
            y_pen = theta_0 * np.cos(omega_n_pen * t_pen)

            fig_pen, ax_pen = plt.subplots()
            ax_pen.plot(t_pen, y_pen, label=f'Ângulo (θ(t))')
            ax_pen.set_title(f"Resposta Não-Amortecida (Oscilador Harmônico)")
            ax_pen.set_xlabel("Tempo (s)")
            ax_pen.set_ylabel("Ângulo (rad)")
            ax_pen.set_ylim(-0.25, 0.25) # Eixo Y fixo
            ax_pen.legend()
            ax_pen.grid(True)
            st.pyplot(fig_pen)
            plt.close(fig_pen)


    elif tipo_sistema == "Sistemas Eletromecânicos":
        st.header("Sistemas Eletromecânicos")
        st.info(r"""
        **Caixa de Ferramentas para Sistemas Eletromecânicos:**
        Estes sistemas acoplam dois domínios físicos. Usamos as leis de ambos:
        * **Domínio Elétrico:** Leis de Kirchhoff (LKT, LKC).
        * **Domínio Mecânico:** Leis de Newton (Translacional ou Rotacional).
        * **Equações de Acoplamento:** Leis que convertem energia elétrica em mecânica e vice-versa (ex: Leis de Lorentz, Faraday).
        """)
        
        with st.expander("Exemplo 1: Motor DC"):
            st.markdown("Vamos modelar a velocidade angular de saída ($\omega$) de um motor DC em resposta a uma tensão de entrada ($E(t)$).")
            
            st.subheader("1. Domínio Mecânico (Lei de Newton-Euller)")
            st.markdown("O somatório dos torques no rotor é igual ao momento de inércia ($J$) vezes a aceleração angular ($\ddot{\theta}$ ou $\dot{\omega}$). O torque resultante ($\tau_R$) é o torque gerado pelo motor ($\tau_g$) menos o torque de atrito ($\tau_f$).")
            st.latex(r"(1) \quad J \frac{d^2\theta}{dt^2} = \tau_R = \tau_g(t) - \tau_f(t)")
            
            st.subheader("2. Domínio Elétrico (Lei das Malhas)")
            st.markdown("A Lei das Malhas de Kirchhoff (LKT) no circuito de armadura é a tensão da fonte ($E$) sendo igual à queda no resistor ($V_R$) mais a Força Contra-Eletromotriz ($V_{emf}$).")
            st.latex(r"(2) \quad E(t) = V_R(t) + V_{emf}(t)")

            st.subheader("3. Relações Constitutivas (Acoplamento)")
            st.markdown("Assumindo **fluxo magnético constante** e **perdas por atrito viscoso**, as equações que conectam os domínios são:")
            st.latex(r"(3) \quad \tau_g(t) = K_1 i(t) \quad (\text{Torque gerado})")
            st.latex(r"(4) \quad V_{emf}(t) = K_2 \omega(t) \quad (\text{Força Contra-Eletromotriz})")
            st.latex(r"(5) \quad \tau_f(t) = B \omega(t) \quad (\text{Torque de Atrito Viscoso})")
            st.markdown("E também temos as relações básicas:")
            st.latex(r"V_R(t) = R \cdot i(t) \quad | \quad \omega(t) = \frac{d\theta}{dt}")
            
            st.subheader("4. Obtenção da Equação Resultante")
            st.markdown("Nosso objetivo é uma EDO para o sistema. Primeiro, isolamos $i(t)$ da equação elétrica (2):")
            st.latex(r"E(t) = R \cdot i(t) + K_2 \omega(t) \implies i(t) = \frac{E(t) - K_2 \omega(t)}{R}")
            st.markdown("Agora, substituímos $i(t)$ (na eq. 3) e $\tau_f$ (eq. 5) na equação mecânica (1):")
            st.latex(r"J \frac{d^2\theta}{dt^2} = \tau_g - \tau_f = K_1 \cdot i(t) - B \omega(t)")
            st.latex(r"J \frac{d^2\theta}{dt^2} = K_1 \left( \frac{E(t) - K_2 \omega(t)}{R} \right) - B \omega(t)")
            st.markdown("Distribuindo os termos, chegamos à equação resultante (6) do slide:")
            st.latex(r"\boxed{J\frac{d^{2}\theta(t)}{dt^{2}} = \frac{K_1}{R}E(t) - \left(\frac{K_1 K_2}{R} + B\right)\omega(t)}")
            st.markdown("Esta é uma EDO de 1ª ordem para a velocidade $\omega(t)$, já que $\\frac{d^2\theta}{dt^2} = \frac{d\omega}{dt}$.")

            st.subheader("5. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Fluxo magnético constante, perdas por atrito viscoso (parâmetros J, B, R, K₁, K₂ constantes), indutância da armadura desprezível.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela tensão $E$), **SISO** (entrada $E$, saída $\omega$), de **Primeira Ordem** (em $\omega$) e **Invariante no Tempo**.
            """)
            
            st.subheader("6. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown("O motor DC se comporta como um sistema de 1ª ordem. Mexa nos parâmetros físicos e veja como eles afetam a velocidade final e o tempo de aceleração.")
            
            col_dc1, col_dc2, col_dc3 = st.columns(3)
            with col_dc1:
                J_dc = st.slider("Inércia (J)", 0.01, 0.5, 0.1, 0.01, key='J_dc')
            with col_dc2:
                B_dc = st.slider("Atrito (B)", 0.01, 0.5, 0.1, 0.01, key='B_dc')
            with col_dc3:
                R_dc = st.slider("Resistência (R)", 0.5, 5.0, 1.0, 0.1, key='R_dc')
            
            col_dc4, col_dc5 = st.columns(2)
            with col_dc4:
                K1_dc = st.slider("Const. Torque (K₁)", 0.01, 2.0, 0.1, 0.01, key='K1_dc')
            with col_dc5:
                K2_dc = st.slider("Const. Elétrica (K₂)", 0.01, 2.0, 0.1, 0.01, key='K2_dc')


            st.markdown("**Análise dos Parâmetros:**")
            st.latex(r"\tau_p = \frac{J R}{B R + K_1 K_2} \quad | \quad K_p = \frac{K_1}{B R + K_1 K_2}")
            st.markdown(r"""
            * **Se $J \uparrow$ (mais inércia):** $\tau_p \uparrow$ (motor **mais lento** para acelerar).
            * **Se $B \uparrow$ (mais atrito):** $\tau_p \downarrow$ (acelera mais rápido) e $K_p \downarrow$ (velocidade final **menor**).
            * **Se $R \uparrow$ (mais resistência):** $\tau_p \downarrow$ e $K_p \downarrow$ (velocidade final **menor**).
            """)
            
            # Cálculo dos parâmetros de 1a ordem
            den = (B_dc * R_dc + K1_dc * K2_dc)
            if den < 1e-6: den = 1e-6 # Evitar divisão por zero
            
            tau_p_dc = (J_dc * R_dc) / den
            Kp_dc = K1_dc / den
            
            t_dc = np.linspace(0, 30, 500) # Eixo do tempo fixo
            y_dc = Kp_dc * (1 - np.exp(-t_dc / tau_p_dc)) # Resposta a um degrau de E(t)=1V

            fig_dc, ax_dc = plt.subplots()
            ax_dc.plot(t_dc, y_dc, label=f'Velocidade (ω(t))')
            ax_dc.axhline(Kp_dc, color='red', linestyle='--', label=f'Vel. Final (Kp = {Kp_dc:.2f} rad/s por Volt)')
            ax_dc.axvline(tau_p_dc, color='gray', linestyle='--', label=f'Const. Tempo (τp = {tau_p_dc:.2f} s)')
            ax_dc.set_title("Resposta de Velocidade do Motor DC (1ª Ordem)")
            ax_dc.set_xlabel("Tempo (s)")
            ax_dc.set_ylabel("Velocidade Angular (ω)")
            ax_dc.set_ylim(0, max(Kp_dc * 1.1, 0.1))
            ax_dc.legend()
            ax_dc.grid(True)
            st.pyplot(fig_dc)
            plt.close(fig_dc)

        with st.expander("Exemplo 2: Válvula Solenoide (Não-Linear)"):
            st.markdown("Este é um sistema eletromecânico complexo onde os parâmetros elétricos dependem da posição mecânica.")
            st.subheader("1. Domínio Elétrico (Acoplado)")
            st.markdown("A indutância $L(x)$ depende da posição $x$ do êmbolo:")
            st.latex(r"u(t) = R i(t) + L(x)\frac{di}{dt} + i(t)\frac{dL(x)}{dx}\frac{dx}{dt}")
            
            st.subheader("2. Domínio Mecânico (Acoplado)")
            st.markdown("A força magnética $F_m$ depende da corrente $i$ e da posição $x$:")
            st.latex(r"m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx(t) = F_m(i, x)")
            
            st.subheader("3. Equação de Acoplamento (Não-Linear)")
            st.latex(r"F_m(i, x) = \frac{1}{2}\frac{dL(x)}{dx}i(t)^2")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Atrito viscoso, mola linear, indutância $L$ é uma função não-linear de $x$.
            * **Classificação:** Modelo **Dinâmico**, **Não-Linear**, **Acoplado**, **Forçado** (pela tensão $u$), **MIMO** (entrada $u$, saídas $x$ e $i$), de **Terceira Ordem** (EDOs para $\dot{x}$, $\ddot{x}$ e $\dot{i}$) e **Invariante no Tempo**.
            """)
            st.info("A simulação de sistemas não-lineares acoplados é altamente complexa e requer métodos numéricos avançados.")

# ==============================================================================
# NÓ 3: ANALISAR MODELO (ORGANIZAÇÃO POR ABAS)
# 

# ==============================================================================
# NÓ 4: MODELAGEM TEÓRICA (AULA INTEGRAL)
# ==============================================================================
elif st.session_state.node == 'modelo_teorico':
    st.header("Aula: Modelagem por Princípios Fundamentais")
    st.success("Esta abordagem usa leis da física e química para descrever um processo.")

    st.subheader("1. As Etapas de uma Modelagem Matemática")
    st.markdown("""
    1.  **Definir o Problema e os Objetivos:** Identificar variáveis.
    2.  **Desenhar um Diagrama:** Volume de Controle.
    3.  **Listar as Premissas:** Hipóteses simplificadoras.
    4.  **Aplicar os Princípios Fundamentais:** Massa, Energia, Momento.
    5.  **Derivar a Equação Final:** EDOs.
    6.  **Validar o Modelo:** Comparar com a realidade.
    """)

    st.subheader("2. A Importância das Premissas")
    st.markdown("Exemplos: Mistura perfeita, fluido incompressível, gás ideal, regime isotérmico.")

    st.subheader("3. As Ferramentas Principais: Leis de Conservação")
    st.markdown("**Balanço de Propriedades Escalares (Massa, Energia):**")
    st.latex(r"\text{ACÚMULO} = \text{ENTRADA} - \text{SAÍDA} + \text{GERAÇÃO} - \text{CONSUMO}")
    st.markdown("**Balanço de Propriedades Vetoriais (Força/Momento):**")
    st.latex(r"\sum \vec{F} = m\vec{a} = \frac{d(m\vec{v})}{dt}")
    
    st.divider()
    st.sidebar.title("Estudos de Caso")
    tipo_sis = st.sidebar.radio("Selecione a categoria:", 
        ("Sistemas de Processos", "Sistemas Elétricos", "Sistemas Mecânicos", "Sistemas Eletromecânicos"), key='rad_te_side')
    
    if st.sidebar.button("⬅️ Voltar para 'Criar Modelo'"):
        st.session_state.node = 'criar_modelo'; st.rerun()

    if tipo_sis == "Sistemas de Processos":
        st.header("Sistemas de Processos (Fluidos)")
        with st.expander("Exemplo 1: Tanque de Nível (Integrador)", expanded=True):
            st.latex(r"A \frac{dh(t)}{dt} = Q_{in}(t) - Q_{out}(t)")
            c1, c2, c3 = st.columns(3)
            qi = c1.slider("Qin", 0.0, 5.0, 2.0, key='te_qi_t')
            qo = c2.slider("Qout", 0.0, 5.0, 1.5, key='te_qo_t')
            ar = c3.slider("Área", 0.5, 5.0, 1.0, key='te_ar_t')
            t_te = np.linspace(0, 50, 200); h_te = np.maximum(0, ((qi - qo)/ar) * t_te + 5)
            fig_te, ax_te = plt.subplots(); ax_te.plot(t_te, h_te); st.pyplot(fig_te); plt.close(fig_te)
        with st.expander("Exemplo 2: Vaso de Gás (Não-Linear)"):
            st.latex(r"\frac{dP}{dt} = \left(\frac{R T}{V \cdot MM}\right) (k_1 \sqrt{P_1 - P} - k_2 \sqrt{P - P_2})")
            cv1, cv2 = st.columns(2)
            p1_v = cv1.slider("P1", 50, 150, 100, key='te_p1_v')
            v_v = cv2.slider("Vol", 0.1, 5.0, 1.0, key='te_v_v')
            def ode_g(t, y, p, v): return [((8.314*300)/(v*0.028)) * (0.01*np.sqrt(max(0, p-y[0])) - 0.01*np.sqrt(max(0, y[0]-10)))]
            sol_v = solve_ivp(ode_g, [0, 50], [20.0], t_eval=np.linspace(0, 50, 200), args=(p1_v, v_v))
            fig_v, ax_v = plt.subplots(); ax_v.plot(sol_v.t, sol_v.y[0], color='orange'); st.pyplot(fig_v); plt.close(fig_v)

    elif tipo_sis == "Sistemas Elétricos":
        st.header("Sistemas Elétricos")
        with st.expander("Exemplo 1: Circuito RC", expanded=True):
            st.latex(r"R \frac{dq(t)}{dt} + \frac{1}{C} q(t) = \epsilon(t)")
            ce1, ce2 = st.columns(2)
            r_te = ce1.slider("R", 0.5, 10.0, 2.0, key='te_r_rc')
            c_te = ce2.slider("C", 0.1, 5.0, 1.0, key='te_c_rc')
            t_rc = np.linspace(0, 20, 200); v_rc = 5 * (1 - np.exp(-t_rc/(r_te*c_te)))
            fig_rc, ax_rc = plt.subplots(); ax_rc.plot(t_rc, v_rc, color='purple'); st.pyplot(fig_rc); plt.close(fig_rc)
        with st.expander("Exemplo 2: Circuito RLC (2ª Ordem)"):
            st.latex(r"LC \frac{d^2V_C}{dt^2} + RC \frac{dV_C}{dt} + V_C = \epsilon")
        with st.expander("Exemplo 3: Malhas em Paralelo"):
            st.latex(r"R_{eq} = (\frac{1}{R_1} + \frac{1}{R_2})^{-1}")
        with st.expander("Exemplo 4: Diodo (Não-Linear)"):
            st.latex(r"i_D = I_S (e^{V_D/nV_T} - 1)")

    elif tipo_sis == "Sistemas Mecânicos":
        st.header("Sistemas Mecânicos")
        with st.expander("Exemplo 1: Massa-Mola-Amortecedor"):
            st.latex(r"m \ddot{z} + c \dot{z} + k z = F(t)")
        with st.expander("Exemplo 2: Associação de 2 Massas"):
            st.latex(r"M_1 \ddot{x}_1 + c(\dot{x}_1 - \dot{x}_2) + k(x_1 - x_2) = 0")
        with st.expander("Exemplo 3: Pêndulo (Lagrange)"):
            st.latex(r"mL^2 \ddot{\theta} + mgL \sin\theta = 0")

    elif tipo_sis == "Sistemas Eletromecânicos":
        st.header("Sistemas Eletromecânicos")
        with st.expander("Exemplo 1: Motor DC", expanded=True):
            st.latex(r"J \dot{\omega} = \frac{K_1}{R}E - (\frac{K_1 K_2}{R} + B)\omega")
            md1, md2 = st.columns(2)
            jm = md1.slider("Inércia (J)", 0.01, 0.5, 0.1, key='te_jm_m')
            bm = md2.slider("Atrito (B)", 0.01, 0.5, 0.1, key='te_bm_m')
            t_m = np.linspace(0, 30, 200); w_m = 1.5 * (1 - np.exp(-t_m/(jm/0.1)))
            fig_m, ax_m = plt.subplots(); ax_m.plot(t_m, w_m, color='green'); st.pyplot(fig_m); plt.close(fig_m)
        with st.expander("Exemplo 2: Válvula Solenoide"):
            st.latex(r"F_m = \frac{1}{2} \frac{dL(x)}{dx} i^2")

# ==============================================================================
# NÓ 5: MODELAGEM EMPÍRICA (TUTORIAL COMPLETO)
# ==============================================================================
elif st.session_state.node == 'modelo_empirico':
    st.header("Modelagem Empírica - Identificação de Sistemas")

    st.markdown("""
    ### 1. Modelagem por Caixa Preta: Quando e Por Que Usar?

    A modelagem empírica, também conhecida como **abordagem de caixa preta**, difere da modelagem fenomenológica em um aspecto fundamental: 
    não partimos de leis físicas detalhadas, mas **inferimos o comportamento do sistema diretamente dos dados experimentais**.
    """)

    st.divider()

    # SELEÇÃO DO TIPO DE SISTEMA
    st.subheader("2. Escolha o Tipo de Sistema para Identificação")

    sistema_tipo = st.radio(
        "Selecione o tipo de sistema que deseja identificar:",
        ["Sistema de 1ª Ordem", "Sistema de 2ª Ordem"],
        horizontal=True,
        key="tipo_sistema_identificacao"
    )

    st.divider()

    if sistema_tipo == "Sistema de 1ª Ordem":
        st.markdown("""
        ### 3. Identificação de Sistemas de 1ª Ordem
        
        **Modelo Matemático:**
        """)
        
        st.latex(r"G(s) = \frac{K}{\tau s + 1} \quad \text{ou no domínio do tempo:} \quad y(t) = K \left(1 - e^{-(t-t_0)/\tau}\right)")
        
        col_1ord1, col_1ord2 = st.columns(2)
        
        with col_1ord1:
            st.markdown("""
            #### **Método da Curva de Reação**
            
            **Procedimento Experimental:**
            
            1. **Estabilize** o sistema em regime permanente
            2. **Aplique um degrau** na entrada em $t = t_0$
            3. **Meça** a resposta $y(t)$
            4. **Extraia** os parâmetros:
            - $K = \\frac{\Delta y_{\infty}}{\Delta u}$
            - $\\tau$ = tempo para atingir 63.2% da variação total
            
            **Parâmetro Ganho (K):**
            $$K = \\frac{y_{final} - y_{inicial}}{u_{final} - u_{inicial}}$$
            
            Representa a **sensibilidade estática** do sistema.
            """)
        
        with col_1ord2:
            st.markdown("""
            #### **Interpretação dos Parâmetros**
            
            **Constante de Tempo (τ):**
            - **63.2%** em $t = t_0 + \\tau$
            - **95.0%** em $t = t_0 + 3\\tau$
            - **98.2%** em $t = t_0 + 4\\tau$ (tempo de acomodação)
            
            **Exemplos Práticos:**
            
            | Sistema Físico | K representa | τ representa |
            |----------------|--------------|--------------|
            | Circuito RC | Ganho estático | R × C |
            | Tanque de Nível | Sensibilidade nível/vazão | A × R (área × resistência) |
            | Sistema Térmico | Sensibilidade temperatura/potência | Capacitância térmica × Resistência |
            
            **Característica Importante:** Sistemas de 1ª ordem têm resposta **exponencial pura**, sem oscilações.
            """)
        
        # Visualização do método para 1ª ordem
        st.markdown("#### **Visualização do Método:**")
        
        fig_1ord_metodo, ax_1ord = plt.subplots(figsize=(10, 5))
        
        # Gerar curva de exemplo
        t_exemplo = np.linspace(0, 15, 300)
        K_ex = 2.5
        tau_ex = 3.0
        t0_ex = 2.0
        
        y_exemplo = np.where(t_exemplo < t0_ex, 0, 
                            K_ex * (1 - np.exp(-(t_exemplo - t0_ex)/tau_ex)))
        
        # Plot
        ax_1ord.plot(t_exemplo, y_exemplo, 'b-', linewidth=2.5, label='Resposta do sistema')
        
        # Marcações
        # Linha de K
        ax_1ord.axhline(K_ex, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_1ord.annotate(f'$K = {K_ex}$', xy=(14, K_ex), xytext=(10, K_ex + 0.3),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
        
        # Linha de 63.2%
        y_63 = 0.632 * K_ex
        t_63 = t0_ex + tau_ex
        ax_1ord.axhline(y_63, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        ax_1ord.axvline(t_63, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        ax_1ord.plot(t_63, y_63, 'go', markersize=8)
        
        ax_1ord.annotate(f'$\\tau = {tau_ex}$ s\n(63.2% em $t_0 + \\tau$)', 
                        xy=(t_63, y_63), xytext=(t_63 + 1, y_63 - 0.5),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')
        
        # Configurações
        ax_1ord.set_xlabel('Tempo (s)', fontsize=11)
        ax_1ord.set_ylabel('Resposta $y(t)$', fontsize=11)
        ax_1ord.set_title('Método da Curva de Reação - Sistema de 1ª Ordem', fontsize=12)
        ax_1ord.grid(True, alpha=0.3)
        ax_1ord.legend(['Resposta ao degrau'], loc='lower right')
        
        st.pyplot(fig_1ord_metodo)
        
        st.markdown("""
        **Dicas Práticas:**
        1. Realize múltiplas medições e calcule a média
        2. Filtre o ruído de alta frequência se necessário
        3. Valide com um segundo experimento independente
        4. Para sistemas com tempo morto, adicione o parâmetro θ (atraso)
        """)

    else:  # Sistema de 2ª Ordem
        st.markdown("""
        ### 3. Identificação de Sistemas de 2ª Ordem
        
        **Modelo Matemático:**
        """)
        
        st.latex(r"G(s) = \frac{K\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}")
        
        col_2ord1, col_2ord2 = st.columns(2)
        
        with col_2ord1:
            st.markdown("""
            #### **Método de Identificação**
            
            **Parâmetros a Identificar:**
            1. **Ganho (K):** Sensibilidade estática
            2. **Frequência Natural (ωₙ):** Frequência das oscilações
            3. **Fator de Amortecimento (ζ):** Grau de amortecimento
            
            **Procedimento:**
            
            1. **Aplicar degrau** e medir resposta
            2. **Medir overshoot (Mₚ):** 
            $$M_p = \\frac{y_{max} - y_{final}}{y_{final}} \\times 100\%$$
            3. **Calcular ζ:** 
            $$\zeta = \sqrt{\\frac{(\ln(M_p/100))^2}{\pi^2 + (\ln(M_p/100))^2}}$$
            4. **Medir tempo de pico (Tₚ):** Tempo do primeiro pico
            5. **Calcular ωₙ:** 
            $$\omega_n = \\frac{\pi}{T_p\sqrt{1-\zeta^2}}$$
            6. **Determinar K:** 
            $$K = \\frac{y_{final}}{\Delta u}$$
            """)
        
        with col_2ord2:
            st.markdown("""
            #### **Interpretação dos Parâmetros**
            
            **Fator de Amortecimento (ζ):**
            - **ζ < 0:** Sistema instável
            - **ζ = 0:** Não amortecido (oscilações contínuas)
            - **0 < ζ < 1:** Subamortecido (oscilações decrescentes)
            - **ζ = 1:** Criticamente amortecido (resposta mais rápida sem overshoot)
            - **ζ > 1:** Superamortecido (sem oscilações)
            
            **Frequência Natural (ωₙ):**
            - Frequência das oscilações naturais do sistema
            - Relacionada à velocidade de resposta
            - ωₙ alto = resposta rápida
            
            **Exemplos Práticos:**
            
            | Sistema Físico | ζ representa | ωₙ representa |
            |----------------|--------------|---------------|
            | Sistema Mola-Massa | Amortecimento | $\sqrt{k/m}$ |
            | Circuito RLC | $R/(2\sqrt{L/C})$ | $1/\sqrt{LC}$ |
            | Suspensão Veicular | Amortecimento do amortecedor | Rigidez da mola/massa |
            
            **Característica:** Sistemas de 2ª ordem podem apresentar **overshoot** e **oscilações**.
            """)
        
        # Visualização do método para 2ª ordem
        st.markdown("#### **Visualização do Método:**")
        
        fig_2ord_metodo, ax_2ord = plt.subplots(figsize=(10, 5))
        
        # Gerar curva de exemplo (subamortecido)
        t_exemplo = np.linspace(0, 15, 500)
        K_ex = 1.0
        zeta_ex = 0.4
        wn_ex = 2.0
        
        # Resposta de 2ª ordem subamortecida
        wd_ex = wn_ex * np.sqrt(1 - zeta_ex**2)
        phi_ex = np.arccos(zeta_ex)
        y_exemplo = K_ex * (1 - (np.exp(-zeta_ex * wn_ex * t_exemplo) / 
                            np.sqrt(1 - zeta_ex**2)) * 
                            np.sin(wd_ex * t_exemplo + phi_ex))
        
        # Plot
        ax_2ord.plot(t_exemplo, y_exemplo, 'b-', linewidth=2.5, label='Resposta do sistema')
        
        # Encontrar overshoot e tempo de pico
        y_max = np.max(y_exemplo)
        t_max_idx = np.argmax(y_exemplo)
        t_max = t_exemplo[t_max_idx]
        
        # Linha do valor final
        ax_2ord.axhline(K_ex, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_2ord.annotate(f'$K = {K_ex}$', xy=(14, K_ex), xytext=(10, K_ex + 0.1),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
        
        # Marcar overshoot
        if y_max > K_ex:
            ax_2ord.plot(t_max, y_max, 'ro', markersize=8)
            overshoot_val = (y_max - K_ex) / K_ex * 100
            
            ax_2ord.annotate(f'Overshoot = {overshoot_val:.1f}%\n$T_p = {t_max:.2f}$ s', 
                            xy=(t_max, y_max), xytext=(t_max + 1, y_max),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontsize=10, color='red')
            
            # Linha de overshoot
            ax_2ord.plot([t_max, t_max], [K_ex, y_max], 'r:', alpha=0.5)
        
        # Marcar período de oscilação
        # Encontrar segundo cruzamento por zero após o pico
        t_search = t_exemplo[t_max_idx:]
        y_search = y_exemplo[t_max_idx:]
        
        # Encontrar próximo mínimo
        t_half = t_max + np.pi / wd_ex  # Aproximação para meio período
        
        ax_2ord.axvline(t_half, color='green', linestyle=':', alpha=0.7, linewidth=1)
        ax_2ord.annotate(f'$T_d = {np.pi/wd_ex:.2f}$ s\n(meio período)', 
                        xy=(t_half, 0.5), xytext=(t_half + 0.5, 0.3),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=9, color='green')
        
        # Configurações
        ax_2ord.set_xlabel('Tempo (s)', fontsize=11)
        ax_2ord.set_ylabel('Resposta $y(t)$', fontsize=11)
        ax_2ord.set_title('Identificação de Sistema de 2ª Ordem - Parâmetros Chave', fontsize=12)
        ax_2ord.grid(True, alpha=0.3)
        ax_2ord.legend(['Resposta ao degrau'], loc='lower right')
        
        st.pyplot(fig_2ord_metodo)
        
        st.markdown("""
        **Dicas Práticas para Sistemas de 2ª Ordem:**
        1. **Overshoot confiável:** Espere até o primeiro pico para medição precisa
        2. **Período de oscilação:** Meça múltiplos ciclos e tire a média
        3. **Sistemas superamortecidos:** Use método de duas constantes de tempo
        4. **Validação:** Compare com resposta a diferentes amplitudes de degrau
        5. **Ruído:** Filtre cuidadosamente para não confundir ruído com oscilações reais
        """)

    st.divider()

    # LABORATÓRIO VIRTUAL COMUM
    st.subheader("4. Laboratório Virtual: Identificação Prática")

    st.markdown(f"""
    **Objetivo:** Identificar os parâmetros de um sistema {sistema_tipo.lower()} desconhecido utilizando dados experimentais.

    **Contexto:** Você tem acesso a dados experimentais de um sistema real. Sua tarefa é ajustar os parâmetros do modelo para minimizar o erro entre previsão e dados observados.
    """)

    # Configuração baseada no tipo de sistema
    if sistema_tipo == "Sistema de 1ª Ordem":
        # Sistema "real" de 1ª ordem
        K_real, tau_real, t0 = 2.7, 3.5, 2.0
        
        # Gerar dados
        t_l = np.linspace(0, 30, 200)
        u_l = np.where(t_l >= t0, 1.0, 0.0)
        
        np.random.seed(42)
        y_clean = np.where(t_l < t0, 0, K_real * (1 - np.exp(-(t_l - t0)/tau_real)))
        y_noisy = y_clean + np.random.normal(0, 0.08, size=len(t_l))
        
        # Layout
        c_m, c_p = st.columns([1, 2])
        
        with c_m:
            st.markdown("#### Ajuste dos Parâmetros")
            
            st.latex(r"G(s) = \frac{K}{\tau s + 1}")
            
            # Controles
            km = st.slider("Ganho (K)", 0.5, 5.0, 1.0, 0.1, 
                        key='sl_k_1ord',
                        help="Variação final na saída para degrau unitário")
            
            tm = st.slider("Constante de tempo τ (s)", 0.5, 10.0, 2.0, 0.1,
                        key='sl_tau_1ord',
                        help="Tempo para atingir 63.2% da resposta")
            
            # Calcular resposta do modelo
            y_m = np.where(t_l < t0, 0, km * (1 - np.exp(-(t_l - t0)/tm)))
            
            # Métricas
            mse_m = np.mean((y_noisy - y_m)**2)
            mae_m = np.mean(np.abs(y_noisy - y_m))
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("MSE", f"{mse_m:.4f}")
            with col_metric2:
                st.metric("MAE", f"{mae_m:.4f}")
            
            # Otimização
            if st.button("Otimização Automática", key='btn_opt_1ord'):
                def obj_1ord(p):
                    K_opt, tau_opt = p
                    y_pred = np.where(t_l < t0, 0, K_opt * (1 - np.exp(-(t_l - t0)/tau_opt)))
                    return np.mean((y_noisy - y_pred)**2)
                
                res = minimize(obj_1ord, [km, tm], bounds=[(0.1, 5.0), (0.1, 10.0)])
                
                if res.success:
                    st.session_state.opt_res_1ord = {
                        "K": res.x[0], 
                        "tau": res.x[1], 
                        "mse": res.fun,
                        "optimized": True
                    }
            
            # Resultados otimizados
            if 'opt_res_1ord' in st.session_state and st.session_state.opt_res_1ord.get("optimized", False):
                st.markdown("---")
                st.markdown("**Resultados Otimizados:**")
                st.markdown(f"- K = {st.session_state.opt_res_1ord['K']:.3f}")
                st.markdown(f"- τ = {st.session_state.opt_res_1ord['tau']:.3f} s")
                st.metric("MSE mínimo", f"{st.session_state.opt_res_1ord['mse']:.5f}")
        
        with c_p:
            # Visualização
            st.markdown("#### Visualização do Ajuste")
            
            fig_lab, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                            gridspec_kw={'height_ratios': [1, 3]},
                                            sharex=True)
            
            # Entrada
            ax1.step(t_l, u_l, 'k-', linewidth=2)
            ax1.set_ylabel("Entrada u(t)", fontsize=11)
            ax1.set_title(f"Identificação de Sistema de 1ª Ordem", fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Saída
            ax2.scatter(t_l, y_noisy, color='blue', s=10, alpha=0.4, label='Dados experimentais')
            ax2.plot(t_l, y_m, 'r-', linewidth=2.5, label=f'Seu modelo: K={km:.2f}, τ={tm:.2f}s')
            ax2.plot(t_l, y_clean, 'gray', alpha=0.5, linewidth=1.5, label='Sistema real', linestyle=':')
            
            if 'opt_res_1ord' in st.session_state and st.session_state.opt_res_1ord.get("optimized", False):
                y_opt = np.where(t_l < t0, 0, 
                            st.session_state.opt_res_1ord["K"] * 
                            (1 - np.exp(-(t_l - t0)/st.session_state.opt_res_1ord["tau"])))
                ax2.plot(t_l, y_opt, 'g--', linewidth=2, label='Otimizado', alpha=0.8)
            
            ax2.set_xlabel('Tempo (s)', fontsize=11)
            ax2.set_ylabel('Saída y(t)', fontsize=11)
            ax2.legend(loc='lower right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_lab)

    else:  # Sistema de 2ª Ordem
        # Sistema "real" de 2ª ordem (subamortecido)
        K_real_2ord = 1.0
        zeta_real = 0.3
        wn_real = 2.5
        t0_2ord = 1.0
        
        # Gerar dados
        t_l = np.linspace(0, 15, 400)
        u_l = np.where(t_l >= t0_2ord, 1.0, 0.0)
        
        # Resposta teórica
        wd_real = wn_real * np.sqrt(1 - zeta_real**2)
        phi_real = np.arccos(zeta_real)
        
        np.random.seed(42)
        y_clean = K_real_2ord * (1 - (np.exp(-zeta_real * wn_real * (t_l - t0_2ord)) / 
                                    np.sqrt(1 - zeta_real**2)) * 
                                    np.sin(wd_real * (t_l - t0_2ord) + phi_real))
        y_clean[t_l < t0_2ord] = 0
        
        # Adicionar ruído
        y_noisy = y_clean + np.random.normal(0, 0.03, size=len(t_l))
        
        # Layout
        c_m, c_p = st.columns([1, 2])
        
        with c_m:
            st.markdown("#### Ajuste dos Parâmetros")
            
            st.latex(r"G(s) = \frac{K\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}")
            
            # Controles
            km_2ord = st.slider("Ganho (K)", 0.5, 2.0, 1.0, 0.1, 
                            key='sl_k_2ord')
            
            zeta_user = st.slider("Fator de amortecimento (ζ)", 0.1, 1.5, 0.5, 0.05,
                                key='sl_zeta_2ord',
                                help="0 < ζ < 1: subamortecido, ζ ≥ 1: superamortecido")
            
            wn_user = st.slider("Frequência natural ωₙ (rad/s)", 0.5, 5.0, 2.0, 0.1,
                            key='sl_wn_2ord')
            
            # Calcular resposta do modelo
            if zeta_user < 1:
                wd_user = wn_user * np.sqrt(1 - zeta_user**2)
                phi_user = np.arccos(zeta_user)
                y_m = km_2ord * (1 - (np.exp(-zeta_user * wn_user * (t_l - t0_2ord)) / 
                                    np.sqrt(1 - zeta_user**2)) * 
                                    np.sin(wd_user * (t_l - t0_2ord) + phi_user))
            elif abs(zeta_user - 1) < 0.001:
                # Criticamente amortecido
                y_m = km_2ord * (1 - (1 + wn_user * (t_l - t0_2ord)) * 
                                np.exp(-wn_user * (t_l - t0_2ord)))
            else:
                # Superamortecido
                r1 = -zeta_user * wn_user + wn_user * np.sqrt(zeta_user**2 - 1)
                r2 = -zeta_user * wn_user - wn_user * np.sqrt(zeta_user**2 - 1)
                y_m = km_2ord * (1 - (r2 * np.exp(r1 * (t_l - t0_2ord)) - 
                                    r1 * np.exp(r2 * (t_l - t0_2ord))) / (r2 - r1))
            
            y_m[t_l < t0_2ord] = 0
            
            # Métricas
            mse_m = np.mean((y_noisy - y_m)**2)
            
            # Calcular overshoot do modelo
            y_max_model = np.max(y_m)
            overshoot_model = max(0, (y_max_model - km_2ord) / km_2ord * 100)
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("MSE", f"{mse_m:.5f}")
            with col_metric2:
                st.metric("Overshoot modelo", f"{overshoot_model:.1f}%" if zeta_user < 1 else "0%")
            
            # Estimar overshoot dos dados (aproximado)
            y_max_data = np.max(y_noisy[t_l > t0_2ord + 0.5])
            overshoot_data = max(0, (y_max_data - K_real_2ord) / K_real_2ord * 100)
            st.metric("Overshoot observado", f"{overshoot_data:.1f}%")
            
            # Otimização
            if st.button("Otimização Automática", key='btn_opt_2ord'):
                def obj_2ord(p):
                    K_opt, zeta_opt, wn_opt = p
                    
                    if zeta_opt < 1:
                        wd_opt = wn_opt * np.sqrt(1 - zeta_opt**2)
                        phi_opt = np.arccos(zeta_opt)
                        y_pred = K_opt * (1 - (np.exp(-zeta_opt * wn_opt * (t_l - t0_2ord)) / 
                                            np.sqrt(1 - zeta_opt**2)) * 
                                            np.sin(wd_opt * (t_l - t0_2ord) + phi_opt))
                    else:
                        # Super/crítico
                        y_pred = np.zeros_like(t_l)
                        for i, t_val in enumerate(t_l):
                            if t_val >= t0_2ord:
                                if abs(zeta_opt - 1) < 0.001:
                                    y_pred[i] = K_opt * (1 - (1 + wn_opt * (t_val - t0_2ord)) * 
                                                    np.exp(-wn_opt * (t_val - t0_2ord)))
                                else:
                                    r1 = -zeta_opt * wn_opt + wn_opt * np.sqrt(zeta_opt**2 - 1)
                                    r2 = -zeta_opt * wn_opt - wn_opt * np.sqrt(zeta_opt**2 - 1)
                                    y_pred[i] = K_opt * (1 - (r2 * np.exp(r1 * (t_val - t0_2ord)) - 
                                                        r1 * np.exp(r2 * (t_val - t0_2ord))) / (r2 - r1))
                    
                    y_pred[t_l < t0_2ord] = 0
                    return np.mean((y_noisy - y_pred)**2)
                
                res = minimize(obj_2ord, [km_2ord, zeta_user, wn_user], 
                            bounds=[(0.1, 2.0), (0.05, 2.0), (0.5, 5.0)])
                
                if res.success:
                    st.session_state.opt_res_2ord = {
                        "K": res.x[0], 
                        "zeta": res.x[1], 
                        "wn": res.x[2],
                        "mse": res.fun,
                        "optimized": True
                    }
            
            # Resultados otimizados
            if 'opt_res_2ord' in st.session_state and st.session_state.opt_res_2ord.get("optimized", False):
                st.markdown("---")
                st.markdown("**Resultados Otimizados:**")
                st.markdown(f"- K = {st.session_state.opt_res_2ord['K']:.3f}")
                st.markdown(f"- ζ = {st.session_state.opt_res_2ord['zeta']:.3f}")
                st.markdown(f"- ωₙ = {st.session_state.opt_res_2ord['wn']:.3f} rad/s")
                st.metric("MSE mínimo", f"{st.session_state.opt_res_2ord['mse']:.6f}")
        
        with c_p:
            # Visualização
            st.markdown("#### Visualização do Ajuste")
            
            fig_lab_2ord, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                                gridspec_kw={'height_ratios': [1, 3]},
                                                sharex=True)
            
            # Entrada
            ax1.step(t_l, u_l, 'k-', linewidth=2)
            ax1.set_ylabel("Entrada u(t)", fontsize=11)
            ax1.set_title(f"Identificação de Sistema de 2ª Ordem", fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Saída
            ax2.scatter(t_l, y_noisy, color='blue', s=8, alpha=0.4, label='Dados experimentais')
            ax2.plot(t_l, y_m, 'r-', linewidth=2.5, 
                    label=f'Seu modelo: K={km_2ord:.2f}, ζ={zeta_user:.2f}, ωₙ={wn_user:.2f}')
            ax2.plot(t_l, y_clean, 'gray', alpha=0.5, linewidth=1.5, 
                    label='Sistema real', linestyle=':')
            
            if 'opt_res_2ord' in st.session_state and st.session_state.opt_res_2ord.get("optimized", False):
                # Calcular resposta otimizada
                K_opt = st.session_state.opt_res_2ord['K']
                zeta_opt = st.session_state.opt_res_2ord['zeta']
                wn_opt = st.session_state.opt_res_2ord['wn']
                
                if zeta_opt < 1:
                    wd_opt = wn_opt * np.sqrt(1 - zeta_opt**2)
                    phi_opt = np.arccos(zeta_opt)
                    y_opt = K_opt * (1 - (np.exp(-zeta_opt * wn_opt * (t_l - t0_2ord)) / 
                                        np.sqrt(1 - zeta_opt**2)) * 
                                        np.sin(wd_opt * (t_l - t0_2ord) + phi_opt))
                else:
                    y_opt = np.zeros_like(t_l)
                    for i, t_val in enumerate(t_l):
                        if t_val >= t0_2ord:
                            if abs(zeta_opt - 1) < 0.001:
                                y_opt[i] = K_opt * (1 - (1 + wn_opt * (t_val - t0_2ord)) * 
                                                np.exp(-wn_opt * (t_val - t0_2ord)))
                            else:
                                r1 = -zeta_opt * wn_opt + wn_opt * np.sqrt(zeta_opt**2 - 1)
                                r2 = -zeta_opt * wn_opt - wn_opt * np.sqrt(zeta_opt**2 - 1)
                                y_opt[i] = K_opt * (1 - (r2 * np.exp(r1 * (t_val - t0_2ord)) - 
                                                    r1 * np.exp(r2 * (t_val - t0_2ord))) / (r2 - r1))
                
                y_opt[t_l < t0_2ord] = 0
                ax2.plot(t_l, y_opt, 'g--', linewidth=2, label='Otimizado', alpha=0.8)
            
            ax2.set_xlabel('Tempo (s)', fontsize=11)
            ax2.set_ylabel('Saída y(t)', fontsize=11)
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Destacar overshoot nos dados
            y_max_data = np.max(y_noisy[t_l > t0_2ord + 0.5])
            idx_max = np.argmax(y_noisy[t_l > t0_2ord + 0.5]) + np.sum(t_l <= t0_2ord + 0.5)
            if y_max_data > K_real_2ord:
                ax2.plot(t_l[idx_max], y_max_data, 'ro', markersize=8)
                ax2.annotate(f'Overshoot observado\n{y_max_data-K_real_2ord:.2f}',
                            xy=(t_l[idx_max], y_max_data),
                            xytext=(t_l[idx_max]+0.5, y_max_data+0.1),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig_lab_2ord)

    st.divider()

    # CONCLUSÕES E NAVEGAÇÃO
    st.markdown("""
    ### 5. Conclusões e Próximos Passos

    **O que aprendemos neste laboratório?**

    1. **Identificação prática:** Como ajustar modelos a dados experimentais
    2. **Importância dos dados:** A qualidade dos dados afeta diretamente a precisão da identificação
    3. **Técnicas de otimização:** Como algoritmos podem ajudar a encontrar os melhores parâmetros
    4. **Validação visual:** A importância de visualizar o ajuste entre modelo e dados

    **Próximos passos recomendados:**

    1. **Análise da resposta:** Use as ferramentas de análise para entender melhor o comportamento do sistema identificado
    2. **Experimentos variados:** Tente identificar sistemas com diferentes características (mais/menos amortecidos, com tempo morto, etc.)
    3. **Validação cruzada:** Teste o modelo em condições diferentes das usadas para identificação
    4. **Aplicação prática:** Use o modelo identificado para sintonia de controladores
    """)

    col_nav1, col_nav2 = st.columns(2)

    with col_nav1:
        if st.button("📈 Análise da Resposta Temporal", 
                    key='btn_nav_analise',
                    use_container_width=True):
            st.session_state.node = 'analise_resposta'
            st.rerun()

    with col_nav2:
        if st.button("⬅️ Voltar ao Menu Principal", 
                    key='btn_voltar_menu',
                    use_container_width=True):
            st.session_state.node = 'criar_modelo'
            st.rerun()

    st.markdown("""
    ---
    **Dica final:** A modelagem empírica é uma ferramenta poderosa, mas lembre-se que os modelos são válidos apenas 
    dentro da faixa de dados utilizados. Sempre valide em condições operacionais diferentes e esteja preparado para 
    reajustar os parâmetros se as condições do processo mudarem.
    """)
