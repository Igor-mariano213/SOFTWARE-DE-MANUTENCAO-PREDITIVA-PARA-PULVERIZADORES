import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Configuração da Página
st.set_page_config(page_title="SmartSpray AI - Detector de Anomalias", layout="wide")

# --- CARREGAMENTO DOS MODELOS E DADOS ---
@st.cache_resource
def carregar_ia():
    # Verifica se os modelos existem
    arquivos = ['modelo_classificador.joblib', 'modelo_regressor.joblib', 
                'encoder_secao.joblib', 'encoder_estado.joblib']
    
    for arq in arquivos:
        if not os.path.exists(arq):
            return None
            
    clf = joblib.load('modelo_classificador.joblib')
    reg = joblib.load('modelo_regressor.joblib')
    le_secao = joblib.load('encoder_secao.joblib')
    le_estado = joblib.load('encoder_estado.joblib')
    return clf, reg, le_secao, le_estado

@st.cache_data
def carregar_dados():
    if os.path.exists('dados_pulverizador_sinteticos.csv'):
        df = pd.read_csv('dados_pulverizador_sinteticos.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

# Inicialização
ia_components = carregar_ia()
df = carregar_dados()

st.title("🚜 SmartSpray: Sistema de Manutenção Preditiva")
st.markdown("**Monitoramento de Saúde e Vida Útil (RUL)**")

if df is not None and ia_components is not None:
    clf, reg, le_secao, le_estado = ia_components
    
    # --- BARRA LATERAL ---
    st.sidebar.header("🕹️ Controle de Simulação")
    equip_id = st.sidebar.selectbox("Equipamento", df['equipamento_id'].unique())
    secao = st.sidebar.radio("Seção da Barra", sorted(df['secao'].unique()))
    
    # Filtragem
    df_filtrado = df[(df['equipamento_id'] == equip_id) & (df['secao'] == secao)].reset_index(drop=True)
    
    # Controle de Tempo
    if 'tempo_horas' not in st.session_state:
        st.session_state['tempo_horas'] = 10.0
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Inteligência Artificial")

    # Botão de Escanear
    if st.sidebar.button("🔍 Escanear Próxima Falha (IA)"):
        with st.spinner('O Random Forest está analisando o futuro...'):
            indice_atual = int(st.session_state['tempo_horas'] * 60)
            dados_futuros = df_filtrado.iloc[indice_atual+1:].copy()
            
            if not dados_futuros.empty:
                # Prepara dados em lote
                dados_futuros['secao_enc'] = le_secao.transform(dados_futuros['secao'])
                dados_futuros['estado_enc'] = le_estado.transform(dados_futuros['estado_operacao'])
                features = ['pressao_bar', 'vazao_L_min', 'temperatura_C', 'setpoint_pressao_bar', 
                            'erro_pressao_bar', 'secao_enc', 'estado_enc']
                
                # Predição
                predicoes = clf.predict(dados_futuros[features])
                indices_anomalia = np.where(predicoes == 1)[0]
                
                if len(indices_anomalia) > 0:
                    salto_para = indice_atual + 1 + indices_anomalia[0]
                    st.session_state['tempo_horas'] = salto_para / 60
                    st.toast(f"Anomalia detectada na hora {st.session_state['tempo_horas']:.1f}!", icon="⚠️")
                else:
                    st.info("Nenhuma falha iminente detectada.")

    # Slider
    total_horas = len(df_filtrado) / 60
    tempo_selecionado = st.sidebar.slider("Horas de Operação:", 0.0, total_horas, 
                                          value=st.session_state['tempo_horas'], step=0.5, key='tempo_horas')
    
    # --- PREPARAÇÃO DOS DADOS ---
    indice = min(int(tempo_selecionado * 60), len(df_filtrado)-1)
    linha_atual = df_filtrado.iloc[indice]
    
    secao_cod = le_secao.transform([linha_atual['secao']])[0]
    estado_cod = le_estado.transform([linha_atual['estado_operacao']])[0]
    X_input = pd.DataFrame([[linha_atual['pressao_bar'], linha_atual['vazao_L_min'], linha_atual['temperatura_C'], 
                             linha_atual['setpoint_pressao_bar'], linha_atual['erro_pressao_bar'], secao_cod, estado_cod]], 
                           columns=['pressao_bar', 'vazao_L_min', 'temperatura_C', 'setpoint_pressao_bar', 
                                    'erro_pressao_bar', 'secao_enc', 'estado_enc'])
    
    # PREDIÇÕES AO VIVO
    predicao_status = clf.predict(X_input)[0]
    probabilidade = clf.predict_proba(X_input)[0][1]
    predicao_rul = reg.predict(X_input)[0]
    
    # --- DASHBOARD ---
    st.sidebar.info(f"📅 {linha_atual['timestamp'].strftime('%d/%m %H:%M')}")
    
    # Definindo Limites de Alerta
    LIMITE_CRITICO_RUL = 48.0 # Horas
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Pressão", f"{linha_atual['pressao_bar']:.2f} bar", f"{linha_atual['erro_pressao_bar']:.2f}")
    c2.metric("Vazão", f"{linha_atual['vazao_L_min']:.2f} L/min")
    c3.metric("RUL Estimado", f"{predicao_rul:.1f} h", delta_color="inverse", 
              delta="Crítico" if predicao_rul < LIMITE_CRITICO_RUL else "Normal")
    
    st.divider()
    
    # --- LÓGICA DE DIAGNÓSTICO MELHORADA ---
    col_status, col_graf = st.columns([1, 2])
    
    with col_status:
        st.subheader("Diagnóstico do Modelo")
        
        # PRIORIDADE 1: Falha Ativa (O modelo diz que JÁ quebrou ou está quebrando)
        if predicao_status == 1:
            st.error(f"🔴 ANOMALIA DETECTADA")
            st.write(f"**Confiança:** {probabilidade*100:.1f}%")
            st.write("**Ação:** Parada Imediata para Reparo.")
            if linha_atual['pressao_bar'] > linha_atual['setpoint_pressao_bar']:
                st.caption("Causa: Pressão acima do setpoint (Provável Entupimento)")
            else:
                st.caption("Causa: Perda de pressão (Provável Vazamento)")
        
        # PRIORIDADE 2: Alerta Preventivo (RUL Baixo, mas ainda operando)
        elif predicao_rul < LIMITE_CRITICO_RUL:
            st.warning(f"🟡 ALERTA PREVENTIVO")
            st.write(f"**RUL Baixo:** {predicao_rul:.1f} horas restantes.")
            st.write("**Ação:** Agendar manutenção para o fim do turno.")
            st.progress(min(predicao_rul/168, 1.0), text="Vida Útil Restante")
            
        # PRIORIDADE 3: Tudo Normal
        else:
            st.success("🟢 SISTEMA OPERACIONAL")
            st.write(f"**Saúde:** Equipamento em conformidade.")
            st.caption(f"Sem anomalias nos padrões de vibração/pressão.")

    with col_graf:
        st.subheader("Tendência de Pressão (Últimas 2h)")
        inicio_graf = max(0, indice-120)
        dados_graf = df_filtrado.iloc[inicio_graf:indice][['timestamp', 'pressao_bar', 'setpoint_pressao_bar']].set_index('timestamp')
        st.line_chart(dados_graf)

elif ia_components is None:
    st.error("⚠️ Modelos não encontrados! Execute 'gerar_modelos.py'.")
