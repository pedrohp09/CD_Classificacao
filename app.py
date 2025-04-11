# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score

# Configuração de visualização
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    # Passo 1: Carregamento e Exploração do Dataset
    print("\n=== CARREGAMENTO E EXPLORAÇÃO DO DATASET ===")
    try:
        # Carregar o dataset (substitua pelo caminho correto do seu arquivo)
        df = pd.read_csv('diabetes.csv')
        
        # Visualizar as primeiras linhas
        print("\nPrimeiras linhas do dataset:")
        print(df.head())

        # Informações sobre o dataset
        print("\nInformações do dataset:")
        print(df.info())

        # Estatísticas descritivas
        print("\nEstatísticas descritivas:")
        print(df.describe())

        # Verificar valores nulos
        print("\nValores nulos por coluna:")
        print(df.isnull().sum())

        # Distribuição da variável target
        print("\nDistribuição da variável target (Outcome):")
        print(df['Outcome'].value_counts())

        # Plotar a distribuição da variável target
        sns.countplot(x='Outcome', data=df)
        plt.title('Distribuição das Classes (0: Não Diabético, 1: Diabético)')
        plt.show()

    except FileNotFoundError:
        print("Erro: Arquivo 'diabetes.csv' não encontrado.")
        print("Certifique-se de que o arquivo está no mesmo diretório do script.")
        return

    # Passo 2: Pré-processamento dos Dados
    print("\n=== PRÉ-PROCESSAMENTO DOS DADOS ===")
    # Separar features e target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Padronizar as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Verificar o balanceamento das classes
    print("\nDistribuição no conjunto de treino:")
    print(y_train.value_counts(normalize=True))

    print("\nDistribuição no conjunto de teste:")
    print(y_test.value_counts(normalize=True))

    # Passo 3: Construção e Treinamento do Modelo
    print("\n=== TREINAMENTO DO MODELO ===")
    # Criar e treinar o modelo
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para a classe positiva

    # Passo 4: Validação e Avaliação do Modelo
    print("\n=== AVALIAÇÃO DO MODELO ===")
    # Métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAcurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall/Sensibilidade: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Diabético', 'Diabético'], 
                yticklabels=['Não Diabético', 'Diabético'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Regressão Logística (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

    # Validação cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nAcurácia com validação cruzada (5 folds): {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # Passo 5: Exemplo de previsão com novos dados
    print("\n=== EXEMPLO DE PREVISÃO ===")
    # Criar um exemplo fictício de novos dados (valores médios das features)
    new_data = pd.DataFrame([{
        'Pregnancies': 3,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 30,
        'Insulin': 80,
        'BMI': 25,
        'DiabetesPedigreeFunction': 0.4,
        'Age': 35
    }])

    # Pré-processar os novos dados
    new_data_scaled = scaler.transform(new_data)

    # Fazer previsão
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)

    print(f"\nPrevisão para novos dados: {'Diabético' if prediction[0] == 1 else 'Não Diabético'}")
    print(f"Probabilidades: [Não Diabético: {prediction_proba[0][0]:.4f}, Diabético: {prediction_proba[0][1]:.4f}]")

if __name__ == "__main__":
    main()