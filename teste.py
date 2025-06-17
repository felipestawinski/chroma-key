import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_mascara_e_foreground(imagem):
    """
    Cria uma máscara de chroma key suave e um foreground inicial.

    Args:
        imagem (numpy.ndarray): A imagem de entrada com fundo verde (formato BGR).

    Returns:
        tuple: (foreground, mascara_suave)
               - foreground: Imagem com fundo preto.
               - mascara_suave: Máscara em tons de cinza (0-255).
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    limite_inferior_verde = np.array([35, 60, 60])
    limite_superior_verde = np.array([85, 255, 255])
    
    mascara_binaria = cv2.inRange(hsv, limite_inferior_verde, limite_superior_verde)
    mascara_suave = cv2.GaussianBlur(mascara_binaria, (15, 15), 0)
    
    mascara_invertida_float = (255 - mascara_suave).astype(float) / 255.0
    mascara_3ch = cv2.merge([mascara_invertida_float, mascara_invertida_float, mascara_invertida_float])
    
    foreground = (imagem.astype(float) * mascara_3ch).astype(np.uint8)
    
    return foreground, mascara_suave

def remover_borda_verde(foreground):
    """
    Remove a contaminação de cor verde (green spill) do foreground.

    Args:
        foreground (numpy.ndarray): A imagem do primeiro plano com fundo preto.

    Returns:
        numpy.ndarray: O foreground com a cor das bordas corrigida.
    """
    foreground_limpo = foreground.copy()
    b, g, r = cv2.split(foreground_limpo)
    
    # A condição para despill: onde o canal verde é maior que o vermelho e o azul.
    # Adicionamos a condição b > 0 para garantir que não estamos modificando o fundo preto.
    condicao_despill = (g > r) & (g > b) & (b > 0)
    
    # Para esses pixels, definimos o novo valor do canal verde como a média dos canais R e B.
    # Usar a média (ou o máximo) evita que as bordas fiquem com cores estranhas.
    novo_g = ((r.astype(float) + b.astype(float)) / 2).astype(np.uint8)
    
    # Aplica a correção apenas nos pixels que satisfazem a condição
    g[condicao_despill] = novo_g[condicao_despill]
    
    foreground_limpo = cv2.merge([b, g, r])
    return foreground_limpo

def compor_cena(foreground, background, mascara):
    """
    Compõe o foreground sobre o background usando a máscara como alfa.

    Args:
        foreground (numpy.ndarray): Imagem do primeiro plano (já limpa).
        background (numpy.ndarray): Imagem de fundo.
        mascara (numpy.ndarray): Máscara em tons de cinza (0-255).

    Returns:
        numpy.ndarray: A imagem final composta.
    """
    # Garante que o background tenha as mesmas dimensões do foreground
    h, w, _ = foreground.shape
    background = cv2.resize(background, (w, h))

    # Normaliza a máscara para usar como canal alfa (valores de 0.0 a 1.0)
    alpha = mascara.astype(float) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha]) # Transforma em 3 canais

    # Inverte o alfa para o background
    alpha_inverso = 1.0 - alpha

    # Fórmula de composição alfa: C = Fa*A + Ba*(1-A)
    # C = Cor final, F = Foreground, B = Background, A = Alpha
    parte_foreground = cv2.multiply(foreground.astype(float), alpha)
    parte_background = cv2.multiply(background.astype(float), alpha_inverso)

    resultado_final = cv2.add(parte_foreground, parte_background).astype(np.uint8)
    
    return resultado_final

# --- Exemplo de Uso ---
if __name__ == '__main__':
    try:
        # 1. Carregar imagens
        imagem_original = cv2.imread('img/4.bmp')
        background_novo = cv2.imread('img/back.jpg') # Coloque sua imagem de fundo aqui!
        
        if imagem_original is None:
            raise FileNotFoundError("Erro: Imagem 'pessoa_fundo_verde.jpg' não encontrada.")
        if background_novo is None:
            raise FileNotFoundError("Erro: Imagem de fundo ('praia.jpg') não encontrada.")

        # 2. Criar máscara e foreground inicial
        foreground_inicial, mascara = criar_mascara_e_foreground(imagem_original)

        # 3. Remover a contaminação verde (despill)
        foreground_limpo = remover_borda_verde(foreground_inicial)

        # 4. Compor a cena final
        cena_final = compor_cena(foreground_limpo, background_novo, 255 - mascara)

        # Exibir os resultados
        # plt.figure(figsize=(18, 6))

        # plt.subplot(1, 4, 1)
        # plt.title('Foreground Inicial (com borda verde)')
        # plt.imshow(cv2.cvtColor(foreground_inicial, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 4, 2)
        # plt.title('Foreground Limpo (Despill)')
        # plt.imshow(cv2.cvtColor(foreground_limpo, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 4, 3)
        # plt.title('Novo Background')
        # plt.imshow(cv2.cvtColor(background_novo, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 4, 4)
        # plt.title('Cena Final Composta')
        # plt.imshow(cv2.cvtColor(cena_final, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()
        
        cv2.imshow('Cena final', cena_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Salva o resultado final
        cv2.imwrite('cena_final_composta.jpg', cena_final)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")