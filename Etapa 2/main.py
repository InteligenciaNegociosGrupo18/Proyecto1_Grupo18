from preprocessor import main_preprocessor
from modelo import modelo

import joblib

def main():
    # Llamar al preprocesador completo
    X_train, X_test, y_train, y_test,y_data,X_data,data_t = main_preprocessor()
    modelo(X_train, y_train, X_test, y_test,y_data,X_data,data_t)


if __name__ == "__main__":
    main()