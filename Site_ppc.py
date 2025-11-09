import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap # Importa a biblioteca para corrigir a indentação

# Configuração da página
st.set_page_config(page_title="Guia de Processos", layout="centered")
st.title("Guia Interativo de PPC")
st.write("Navegue pelas perguntas para encontrar o caminho correto para o seu problema. Produto tecnológico feito por Nicole.")


if 'node' not in st.session_state:
    st.session_state.node = 'inicio'

# NÓ 1: INÍCIO
if st.session_state.node == 'inicio':
    st.header("Qual é o seu principal objetivo agora?")
    
    if st.button("Quero criar um modelo matemático para um processo.", use_container_width=True):
        st.session_state.node = 'criar_modelo'
        st.rerun()
        
    if st.button("Eu já tenho um modelo e quero analisá-lo.", use_container_width=True):
        st.session_state.node = 'analisar_modelo'
        st.rerun()

# NÓ 2: CRIAR MODELO (LÓGICA REESTRUTURADA)
elif st.session_state.node == 'criar_modelo':
    st.header("Qual é o seu ponto de partida para a modelagem?")
    st.info("Escolha a opção que melhor descreve sua necessidade atual.")

    if st.button("Preciso aprender o passo a passo do zero (Modelagem Teórica)", use_container_width=True):
        st.session_state.node = 'modelo_teorico'
        st.rerun()
    
    if st.button("Tenho apenas dados de um experimento (Modelagem Empírica)", use_container_width=True):
        st.session_state.node = 'modelo_empirico'
        st.rerun()
        
    if st.button("⬅️ Voltar ao Início", use_container_width=True):
        st.session_state.node = 'inicio'
        st.rerun()

# NÓ 3: ANALISAR MODELO
elif st.session_state.node == 'analisar_modelo':
    st.header("Qual aspecto do modelo você quer investigar?")

    if st.button("Quero saber se o sistema é estável.", use_container_width=True):
        st.session_state.node = 'analise_estabilidade'
        st.rerun()
        
    if st.button("Quero entender como o sistema responde a uma mudança.", use_container_width=True):
        st.session_state.node = 'analise_resposta'
        st.rerun()

    if st.button("⬅️ Voltar ao Início"):
        st.session_state.node = 'inicio'
        st.rerun()

# --- PÁGINAS FINAIS (CONTEÚDO) ---

# PÁGINA: MODELAGEM TEÓRICA (AULA COMPLETA E REESTRUTURADA)
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

    st.subheader("3. A Ferramenta Principal: O Balanço Geral")
    st.markdown("A maioria dos modelos em engenharia de processos nasce da lei de conservação, que pode ser escrita como:")
    st.latex(r"\text{ACÚMULO} = \text{ENTRADA} - \text{SAÍDA} + \text{GERAÇÃO} - \text{CONSUMO}")
    
    st.subheader("4. O Que São as Propriedades de um Modelo?")
    st.markdown("Após derivar um modelo, nós o classificamos para entender sua estrutura matemática e complexidade.")
    
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
        st.header("Sistemas de Processos (Fluidos e Térmicos)")
        st.info(r"""
        **Caixa de Ferramentas para Sistemas de Processos:**
        A ferramenta principal aqui é o **Balanço Geral** (Massa, Energia ou Momento) aplicado a um Volume de Controle.
        $\text{ACÚMULO} = \text{ENTRADA} - \text{SAÍDA} + \text{GERAÇÃO} - \text{CONSUMO}$
        """)
        
        with st.expander("Exemplo 1: Vaso Pulmão (Tanque de Nível) - 1ª Ordem Linear"):
            st.markdown("Este é o exemplo clássico de um sistema de primeira ordem. O objetivo é modelar como a altura **h(t)** do líquido varia com a vazão de entrada **qin(t)**.")
            st.subheader("1. Princípio da Conservação de Massa")
            st.latex(r"A \frac{dh(t)}{dt} = q_{in}(t) - q_{out}(t)")
            st.subheader("2. Relação Constitutiva (Saída)")
            st.latex(r"q_{out}(t) = \frac{h(t)}{R_h}")
            st.subheader("3. Obtenção da EDO")
            st.latex(r"\boxed{AR_h \frac{dh}{dt} + h(t) = R_h q_{in}(t)}")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Área do tanque (A) constante, resistência hidráulica (Rh) linear e constante, fluido incompressível, parâmetros concentrados.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela entrada $q_{in}$), **SISO** (entrada $q_{in}$, saída $h$), de **Primeira Ordem** e **Invariante no Tempo**.
            """)

            st.subheader("5. Prévia da Resposta Dinâmica (Interativo)")
            st.markdown("Como a **Área (A)** e a **Resistência (Rh)** afetam a resposta do tanque a um degrau na vazão de entrada?")
            
            col1, col2 = st.columns(2)
            with col1:
                A_tanque = st.slider("Área do Tanque (A)", 0.5, 5.0, 1.0, 0.5, key='A_tanque')
            with col2:
                Rh_tanque = st.slider("Resistência Hidráulica (Rh)", 0.5, 5.0, 2.0, 0.5, key='Rh_tanque')

            st.markdown("**Análise dos Parâmetros:**")
            st.markdown(r"""
            As duas características principais do sistema são o Ganho ($K_p$) e a Constante de Tempo ($\tau_p$):
            * **Ganho ($K_p = R_h$):** Define o valor final da altura para uma entrada degrau. Se $R_h \uparrow$, o ganho $\uparrow$ (o nível final será mais alto).
            * **Constante de Tempo ($\tau_p = A \cdot R_h$):** Define a "velocidade" do sistema. 
                * Se $A \uparrow$ (tanque mais largo) $\implies \tau_p \uparrow$ (sistema **mais lento**).
                * Se $R_h \uparrow$ (saída mais restrita) $\implies \tau_p \uparrow$ (sistema **mais lento**).
            """)

            Kp_tanque = Rh_tanque
            tau_p_tanque = A_tanque * Rh_tanque
            t_tanque = np.linspace(0, 50, 500) # Eixo do tempo fixo
            y_tanque = Kp_tanque * (1 - np.exp(-t_tanque / tau_p_tanque))

            fig_tanque, ax_tanque = plt.subplots()
            ax_tanque.plot(t_tanque, y_tanque, label=f'Resposta (h(t))')
            ax_tanque.axhline(Kp_tanque, color='red', linestyle='--', label=f'Valor Final (Kp = {Kp_tanque:.2f})')
            ax_tanque.axvline(tau_p_tanque, color='gray', linestyle='--', label=f'Const. Tempo (τp = {tau_p_tanque:.2f} s)')
            ax_tanque.set_title("Resposta ao Degrau de 1ª Ordem")
            ax_tanque.set_xlabel("Tempo (s)")
            ax_tanque.set_ylabel("Altura (h)")
            ax_tanque.set_ylim(0, max(5.5, Kp_tanque * 1.1))
            ax_tanque.legend()
            ax_tanque.grid(True)
            st.pyplot(fig_tanque)
            plt.close(fig_tanque)

        with st.expander("Exemplo 2: Vaso de Gás Pressurizado - Modelo Não-Linear"):
            st.markdown("Este modelo descreve a dinâmica da pressão `P(t)` em um vaso de volume `V`.")
            st.subheader("1. Princípio da Conservação de Massa")
            st.latex(r"\frac{dm(t)}{dt} = F_{entrada}(t) - F_{saida}(t)")
            st.subheader("2. Equações Constitutivas e Premissas")
            st.markdown("""
            * **Premissas:** Gás ideal, isotérmico, volume constante, parâmetros concentrados, aberturas de fluxo (k₁, k₂) constantes.
            * **Fluxos Mássicos (`F`):** Escoamento turbulento (não-linear).
            """)
            st.latex(r"F_{entrada} \propto \sqrt{P_1 - P(t)}, \quad F_{saida} \propto \sqrt{P(t) - P_2}")
            st.markdown("**Relação Massa-Pressão (Acúmulo):** Lei dos Gases Ideais.")
            st.latex(r"m(t) = \left(\frac{V \cdot MM}{R \cdot T}\right) P(t) \implies \frac{dm}{dt} = \left(\frac{V \cdot MM}{R \cdot T}\right) \frac{dP}{dt}")
            
            st.subheader("3. Obtenção do Modelo Dinâmico Final")
            st.latex(r"\boxed{\frac{dP}{dt} \propto \left( k_1 \sqrt{P_1 - P} - k_2 \sqrt{P - P_2} \right)}")
            
            st.subheader("4. Análise e Classificação do Modelo")
            st.markdown("Modelo **Dinâmico**, **Não-Linear** (devido à raiz quadrada), **Forçado** (pelas pressões $P_1, P_2$), **MISO** (entradas $P_1, P_2$, saída $P$), de **Parâmetros Concentrados** e **Invariante no Tempo**.")

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
            st.markdown("Queremos a EDO em $V_C$. Usamos as relações baseadas em $i = C \frac{dV_C}{dt}$:")
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

        with st.expander("Exemplo 4: Sistema Não-Linear (Circuito com Diodo)"):
            st.markdown("Circuitos com componentes semicondutores, como diodos, são exemplos clássicos de sistemas não-lineares.")
            st.subheader("1. O Componente Não-Linear")
            st.markdown("A relação entre corrente (`i_D`) e tensão (`V_D`) em um diodo é exponencial, o que torna o sistema não-linear:")
            st.latex(r"i_D = I_S \left( e^{V_D / (n V_T)} - 1 \right)")
            st.subheader("2. Modelagem (Leis de Kirchhoff)")
            st.latex(r"\text{(Malha): } \epsilon(t) = V_R + V_L + V_C")
            st.latex(r"\text{(Nó): } i = i_C + i_D")
            st.subheader("3. Equações de Estado Finais")
            st.markdown("O modelo resultante é um sistema de EDOs (SEDO) não-lineares, onde $f(V_C)$ é a função não-linear do diodo:")
            st.latex(r"\frac{dV_C}{dt} = \frac{1}{C}(i - f(V_C))")
            st.latex(r"\frac{di}{dt} = \frac{1}{L}(\epsilon(t) - V_C - R \cdot i)")
            
            st.subheader("4. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Componentes (R, L, C) são ideais e constantes. A fonte é ideal. O diodo tem comportamento não-linear conhecido $f(V_C)$.
            * **Classificação:** Modelo **Dinâmico**, **Não-Linear** (devido ao diodo), **Forçado** (pela fonte $\epsilon$), **MIMO** (entrada $\epsilon$, saídas $V_C$ e $i$), de **Segunda Ordem** e **Invariante no Tempo**.
            """)
            st.info("A simulação de sistemas não-lineares requer métodos numéricos e será vista em tópicos futuros.")

    elif tipo_sistema == "Sistemas Mecânicos":
        st.header("Sistemas Mecânicos")
        # --- ATUALIZADO ---
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
            # --- MODELO CORRIGIDO PARA NÃO-FORÇADO (FIEL AO SLIDE) ---
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
            z0 = 1.0 # Condição inicial z(0)=1
            v0 = 0.0 # Condição inicial z'(0)=0

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

        # --- EXEMPLO NOVO: PÊNDULO SIMPLES ---
        with st.expander("Exemplo 3: Pêndulo Simples (Formulação Lagrangiana)"):
            st.markdown("Modelagem do ângulo $\\theta(t)$ de um pêndulo simples de comprimento $L$ e massa $m$.")
            st.subheader("1. Princípio Fundamental (Formulação Lagrangiana)")
            st.markdown("Usamos a abordagem da energia, que é mais simples para sistemas rotacionais. A coordenada generalizada é $q = \theta$.")
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
            st.markdown("""
            * **Premissas:** Haste de massa desprezível, pivô sem atrito, massa pontual, movimento em 2D, ângulos pequenos.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Não-Forçado** (autônomo), **SISO** (sem entrada, saída $\\theta$), de **Segunda Ordem** e **Invariante no Tempo**.
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
            st.markdown("O somatório dos torques no rotor é igual ao momento de inércia ($J$) vezes a aceleração angular ($\ddot{\\theta}$ ou $\dot{\omega}$). O torque resultante ($\tau_R$) é o torque gerado pelo motor ($\tau_g$) menos o torque de atrito ($\tau_f$).")
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
            st.markdown("Esta é uma EDO de 1ª ordem para a velocidade $\omega(t)$, já que $\\frac{d^2\theta}{dt^2} = \\frac{d\omega}{dt}$.")

            st.subheader("5. Premissas e Classificação")
            st.markdown("""
            * **Premissas:** Fluxo magnético constante, perdas por atrito viscoso (parâmetros J, B, R, K₁, K₂ constantes), indutância da armadura desprezível.
            * **Classificação:** Modelo **Dinâmico**, **Linear**, **Forçado** (pela tensão $E$), **SISO** (entrada $E$, saída $\\omega$), de **Primeira Ordem** (em $\\omega$) e **Invariante no Tempo**.
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


# PÁGINA: MODELAGEM EMPÍRICA
elif st.session_state.node == 'modelo_empirico':
    st.header("Você está no caminho da Modelagem Empírica!")
    st.info("💡 Tópico a ser abordado conforme o avanço da disciplina.")
    st.markdown("Nesta abordagem, aprenderemos a criar modelos matemáticos a partir de dados experimentais, sem necessariamente conhecer as equações físicas do processo.")

    if st.button("⬅️ Voltar", key='voltar_empirico'):
        st.session_state.node = 'criar_modelo'
        st.rerun()
        
# PÁGINAS DE ANÁLISE (ADC)
elif st.session_state.node in ['analise_estabilidade', 'analise_resposta']:
    if st.session_state.node == 'analise_estabilidade':
        st.header("Análise de Estabilidade")
    else:
        st.header("Análise da Resposta Dinâmica")

    st.info("💡 Tópico a ser abordado conforme o avanço da disciplina.")
    st.markdown("Após a modelagem, o próximo passo crucial é a análise. Nesta seção, aprenderemos a extrair informações importantes do modelo, como sua estabilidade e comportamento dinâmico.")
    
    if st.button("⬅️ Voltar", key='voltar_analise'):
        st.session_state.node = 'analisar_modelo'
        st.rerun()