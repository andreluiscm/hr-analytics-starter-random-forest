from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skl_metrics


def prepare_dataframe(dataframe:pd.DataFrame, y_col_name:np.str_) -> tuple[pd.DataFrame, dict]:
    df = dataframe.copy()

    columns_to_drop = ['employee_id', 'hire_date', 'term_date', 'term_reason', 'term_type', 'active_status']
    columns_to_drop.remove(y_col_name)
    
    df = df.drop(columns=columns_to_drop)
    df = df.dropna(subset=[y_col_name])

    columns_to_encode = ['department', 'sub-department', 'job_level',
                         'gender', 'sexual_orientation', 'race',
                         'education', 'location', 'location_city',
                         'marital_status', 'employment_status']
    columns_to_encode.append(y_col_name)
    
    encodings = {}

    for col in columns_to_encode:
        encoded_labels, unique_categories = pd.factorize(df[col])
        
        col_encodings = {}
        for encoding, category in enumerate(unique_categories):
            col_encodings[encoding] = category

        df[col] = encoded_labels
        encodings[col] = col_encodings

    return df, encodings


def split_X_y_dataframe(dataframe:pd.DataFrame, y_col_name:np.str_) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    df, encodings = prepare_dataframe(dataframe, y_col_name)

    y = df[y_col_name].to_numpy()
    X = df.drop(columns=[y_col_name]).to_numpy()

    return df, X, y, encodings


def run(X:np.ndarray, y:np.ndarray, k:np.int_=5, n_estimators:np.int_=100, max_depth:np.int_=10) -> tuple[np.ndarray, np.ndarray]:
    kf = StratifiedKFold(n_splits=k, shuffle=True)

    y_test_labels = []
    y_pred_labels = []

    for _, (train_indices, test_indices) in enumerate(kf.split(X, y)):
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        X_test = X[test_indices]
        y_test = y[test_indices]

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, class_weight='balanced')
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)

        y_test_labels = y_test_labels + list(y_test)
        y_pred_labels = y_pred_labels + list(pred)

    y_test_labels = np.array(y_test_labels)
    y_pred_labels = np.array(y_pred_labels)

    return np.array(y_test_labels), np.array(y_pred_labels)


if __name__ == '__main__':
    # Raw data
    df_raw = pd.read_csv(filepath_or_buffer='./data/people_analytics_start.csv', dtype=str)

    # K-fold params
    k = 5
    # Random forest params
    n_estimators = 100
    max_depth = 5

    # 'active_status' column classification
    col_name = 'active_status'
    df, X, y, encodings = split_X_y_dataframe(df_raw, col_name)
    
    num_active_users = len(df.loc[df['active_status'] == 0])
    num_not_active_users = len(df.loc[df['active_status'] == 1])

    print(f'attribute: {col_name}')
    print(f'encoding: {encodings[col_name]}')
    print(f'# active users: {num_active_users}')
    print(f'# not active users: {num_not_active_users}')
    print(f'df:\n{df}\nX:\n{X}\ny:\n{y}')
    
    y_test, y_pred = run(X, y, k, n_estimators, max_depth)
    cm = skl_metrics.confusion_matrix(y_test, y_pred)
    cr = skl_metrics.classification_report(y_test, y_pred)

    cmd = skl_metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ativo', 'Inativo'])
    ax = cmd.plot(cmap='binary')
    # plt.title('Matriz de Confusão')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Valores Verdadeiros')
    plt.tight_layout()
    plt.savefig('./figures/cm_situation.png')
    plt.close()
    
    print(f'confusion matrix:\n{cm}')
    print(f'classification report:\n{cr}\n')

    # 'term_type' column classiication
    col_name = 'term_type'
    df, X, y, encodings = split_X_y_dataframe(df_raw, col_name)
    
    print(f'attribute: {col_name}')
    print(f'encoding: {encodings[col_name]}')
    print(f'df:\n{df}\nX:\n{X}\ny:\n{y}')
    
    y_test, y_pred = run(X, y, k, n_estimators, max_depth)
    cm = skl_metrics.confusion_matrix(y_test, y_pred)
    cr = skl_metrics.classification_report(y_test, y_pred)
    
    print(f'confusion matrix:\n{cm}')
    print(f'classification report:\n{cr}')

    cmd = skl_metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Voluntário', 'Involuntário'])
    ax = cmd.plot(cmap='binary')
    # plt.title('Matriz de Confusão')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Valores Verdadeiros')
    plt.tight_layout()
    plt.savefig('./figures/cm_termination_type.png')
    plt.close()
