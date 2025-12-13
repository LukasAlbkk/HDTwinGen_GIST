#!/usr/bin/env python3
"""
Compara dois datasets para verificar se são 100% iguais
"""
import pandas as pd
import sys

def compare_datasets(file1, file2):
    """Compara dois CSV e retorna se são idênticos."""

    print("="*70)
    print("COMPARAÇÃO DE DATASETS")
    print("="*70)

    # Carregar datasets
    print(f"\n1. Carregando datasets...")
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        print(f"   ✓ Dataset 1: {len(df1)} linhas, {len(df1.columns)} colunas")
        print(f"   ✓ Dataset 2: {len(df2)} linhas, {len(df2.columns)} colunas")
    except Exception as e:
        print(f"   ✗ Erro ao carregar: {e}")
        return False

    # Verificar dimensões
    print(f"\n2. Comparando dimensões...")
    if df1.shape != df2.shape:
        print(f"   ✗ DIFERENTES!")
        print(f"     Dataset 1: {df1.shape}")
        print(f"     Dataset 2: {df2.shape}")
        return False
    print(f"   ✓ Mesmas dimensões: {df1.shape}")

    # Verificar colunas
    print(f"\n3. Comparando colunas...")
    if list(df1.columns) != list(df2.columns):
        print(f"   ✗ DIFERENTES!")
        print(f"     Apenas em Dataset 1: {set(df1.columns) - set(df2.columns)}")
        print(f"     Apenas em Dataset 2: {set(df2.columns) - set(df1.columns)}")
        return False
    print(f"   ✓ Mesmas colunas: {len(df1.columns)}")

    # Comparar valores
    print(f"\n4. Comparando valores...")
    try:
        # Comparação direta
        comparison = df1.equals(df2)
        if comparison:
            print(f"   ✓ DATASETS SÃO 100% IDÊNTICOS!")
            return True

        # Se não são iguais, mostrar diferenças
        print(f"   ✗ DATASETS SÃO DIFERENTES!")

        # Encontrar diferenças
        diff_mask = (df1 != df2)
        diff_count = diff_mask.sum().sum()
        print(f"     Total de células diferentes: {diff_count}")

        # Mostrar algumas diferenças
        for col in df1.columns:
            if diff_mask[col].any():
                n_diff = diff_mask[col].sum()
                print(f"     - Coluna '{col}': {n_diff} valores diferentes")

                # Mostrar exemplo
                idx = diff_mask[col].idxmax()
                print(f"       Exemplo (linha {idx}):")
                print(f"         Dataset 1: {df1.loc[idx, col]}")
                print(f"         Dataset 2: {df2.loc[idx, col]}")
                break

        return False

    except Exception as e:
        print(f"   ✗ Erro na comparação: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python compare_datasets.py <arquivo1.csv> <arquivo2.csv>")
        print("\nExemplo:")
        print("  python compare_datasets.py dataset1.csv dataset2.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    result = compare_datasets(file1, file2)

    print("\n" + "="*70)
    if result:
        print("RESULTADO: DATASETS SÃO IDÊNTICOS ✓")
    else:
        print("RESULTADO: DATASETS SÃO DIFERENTES ✗")
    print("="*70)

    sys.exit(0 if result else 1)
