import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Carregar dados
print("Carregando dados...")
df = pd.read_csv('dados_pulverizador_sinteticos.csv')

# 2. Preparar Encoders (Transformar texto em número)
le_secao = LabelEncoder()
df['secao_enc'] = le_secao.fit_transform(df['secao'])

le_estado = LabelEncoder()
df['estado_enc'] = le_estado.fit_transform(df['estado_operacao'])

# 3. Definir Features e Targets
features = ['pressao_bar', 'vazao_L_min', 'temperatura_C', 'setpoint_pressao_bar', 
            'erro_pressao_bar', 'secao_enc', 'estado_enc']

X = df[features]
y_class = df['manut_prevista']
y_reg = df['RUL_horas']

# 4. Treinar Modelos (Usando todos os dados para o modelo final da demo)
print("Treinando Classificador Random Forest...")
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X, y_class)

print("Treinando Regressor RUL...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y_reg)

# 5. Salvar tudo para usar no App
print("Salvando arquivos .joblib...")
joblib.dump(clf, 'modelo_classificador.joblib')
joblib.dump(reg, 'modelo_regressor.joblib')
joblib.dump(le_secao, 'encoder_secao.joblib')
joblib.dump(le_estado, 'encoder_estado.joblib')

print("✅ SUCESSO! Modelos salvos na pasta.")