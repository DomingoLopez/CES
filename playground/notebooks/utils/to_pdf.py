
import os
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def create_pdf_from_images(path_save_pdf: Path, images: list, lvlm_manual: pd.DataFrame):
    # Creamos el documento con tamaño de página "letter"
    doc = SimpleDocTemplate(path_save_pdf, pagesize=letter)
    story = []
    estilos = getSampleStyleSheet()
    estilo_normal = estilos['Normal']
    estilo_rojo = estilos['Normal'].clone('rojo')
    estilo_rojo.textColor = colors.red
    
    filas = 8  # Número de filas en la cuadrícula
    columnas = 5  # Número de columnas en la cuadrícula
    max_imgs = filas * columnas
    
    tabla_datos = []  # Lista para almacenar filas de la tabla
    fila_actual = []  # Fila actual de la tabla
    
    for i, ruta in enumerate(images):
        img = Image(str(ruta), width=80, height=80)  # Ajustamos tamaño de imagen
        
        # Obtener la categoría de la imagen
        nombre_imagen = Path(ruta).resolve().str.split("/"[-1])
        print(nombre_imagen)
        fila_df = lvlm_manual[lvlm_manual['img'] == nombre_imagen]
        
        if not fila_df.empty:
            category_llava = fila_df.iloc[0]['category_llava']
            manual_category = fila_df.iloc[0]['manual_category']
            
            if category_llava != manual_category:
                caption = Paragraph(f"{category_llava} / {manual_category}", estilo_rojo)
            else:
                caption = Paragraph(f"{category_llava} / {manual_category}", estilo_normal)
        else:
            caption = Paragraph("N/A", estilo_normal)
        
        # Usamos KeepTogether para garantizar que la imagen y el caption estén en la misma celda
        celda = KeepTogether([img, caption])
        fila_actual.append(celda)
        
        # Si la fila está completa, la añadimos a la tabla y creamos una nueva
        if len(fila_actual) == columnas:
            tabla_datos.append(fila_actual)
            fila_actual = []
        
        # Si es la última imagen y la fila no está llena, la completamos con espacios vacíos
        if i == len(images) - 1 and fila_actual:
            while len(fila_actual) < columnas:
                fila_actual.append(Paragraph(" ", estilo_normal))  # Espacio vacío válido
            tabla_datos.append(fila_actual)
    
    # Crear tabla solo si hay datos
    if tabla_datos:
        tabla = Table(tabla_datos, colWidths=[90]*columnas, rowHeights=[100]*filas)
        tabla.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        story.append(tabla)
        story.append(Spacer(1, 12))  # Espacio entre tablas
    
    doc.build(story)






if __name__ == "__main__":
    pass