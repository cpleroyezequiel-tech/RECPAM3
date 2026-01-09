import streamlit as st
import pandas as pd
import io
import numpy as np

# 1. CONFIGURACION DE PAGINA
st.set_page_config(page_title="Reporte Gestion Contable", layout="wide")

st.title("📊 Análisis de Ventas con Variación Real")
st.markdown("""
**Instrucciones:**
1. Defina el **Mes de Inicio** de su ejercicio.
2. Ingrese el **Periodo de Cierre** (formato AAAAMM, ej: 202412).
3. Copie las 3 columnas de su Excel (**Periodo, Compras, Ventas**) y péguelas abajo.
""")

# 2. BASE DE INDICES (2022 - 2025)
indices_base = {
    "2022/01": 605.0317, "2022/02": 633.4341, "2022/03": 676.0566, "2022/04": 716.9399,
    "2022/05": 753.1470, "2022/06": 793.0278, "2022/07": 851.7610, "2022/08": 911.1316,
    "2022/09": 967.3076, "2022/10": 1028.7060, "2022/11": 1079.2787, "2022/12": 1134.5875,
    "2023/01": 1202.9790, "2023/02": 1282.7091, "2023/03": 1381.1601, "2023/04": 1497.2147,
    "2023/05": 1613.5895, "2023/06": 1709.6115, "2023/07": 1818.0838, "2023/08": 2044.2832,
    "2023/09": 2304.9242, "2023/10": 2496.2730, "2023/11": 2816.0628, "2023/12": 3533.1922,
    "2024/01": 4261.5324, "2024/02": 4825.7881, "2024/03": 5357.0929, "2024/04": 5830.2271,
    "2024/05": 6073.7165, "2024/06": 6351.7145, "2024/07": 6607.7479, "2024/08": 6883.4412,
    "2024/09": 7122.2421, "2024/10": 7313.9542, "2024/11": 7491.4314, "2024/12": 7694.0075,
    "2025/01": 7864.1257, "2025/02": 8052.9927, "2025/03": 8353.3125, "2025/04": 8585.6078,
    "2025/05": 8714.4871, "2025/06": 8855.5681, "2025/07": 9023.9730, "2025/08": 9193.2441,
    "2025/09": 9384.0922, "2025/10": 9603.8623, "2025/11": 9841.3581
}

# 3. SELECTORES
col_conf1, col_conf2 = st.columns(2)
with col_conf1:
    meses_nombres = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
                     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    mes_inicio_nombre = st.selectbox("Mes de INICIO del ejercicio:", meses_nombres, index=0)
    mes_inicio_num = meses_nombres.index(mes_inicio_nombre) + 1

with col_conf2:
    raw_input = st.text_input("Reexpresar a moneda de (AAAAMM):", value="202412", max_chars=6)
    if len(raw_input) == 6 and raw_input.isdigit():
        mes_destino_input = f"{raw_input[:4]}/{raw_input[4:]}"
    else:
        mes_destino_input = raw_input

st.divider()

data_pegada = st.text_area("Pegue las 3 columnas aquí (Año-Mes, Compras, Ventas):", height=200)

if data_pegada:
    if len(raw_input) < 6:
        st.warning("Por favor, complete el periodo con 6 dígitos (ejemplo: 202412).")
    elif mes_destino_input not in indices_base:
        st.error("⚠️ No se admiten reexpresiones anteriores al 2022/01 o posteriores al 2025/11.")
    else:
        try:
            df = pd.read_csv(io.StringIO(data_pegada), sep='\t')
            df.columns = ['Periodo_Raw', 'Compras_H', 'Venta_H_Raw']
            
            def normalizar_periodo(val):
                val = str(val).strip()
                if '/' in val:
                    partes = val.split('/')
                    return f"{partes[0]}/{partes[1].zfill(2)}"
                return val

            df['Periodo'] = df['Periodo_Raw'].apply(normalizar_periodo)

            def limpiar_monto(val):
                val = str(val).strip().upper()
                if val in ["S/D", "NAN", ""]: return np.nan
                val = val.replace('$', '').replace(' ', '')
                if '.' in val and ',' in val:
                    if val.rfind('.') < val.rfind(','):
                        val = val.replace('.', '').replace(',', '.')
                    else:
                        val = val.replace(',', '')
                elif ',' in val:
                    if val.count(',') > 1 or len(val.split(',')[-1]) == 3:
                        val = val.replace(',', '')
                    else:
                        val = val.replace(',', '.')
                elif '.' in val:
                    if val.count('.') > 1 or len(val.split('.')[-1]) == 3:
                        val = val.replace('.', '')
                try: return float(val)
                except: return np.nan

            df['Venta_H'] = df['Venta_H_Raw'].apply(limpiar_monto)

            lista_periodos = sorted(list(indices_base.keys()))
            idx_corte = lista_periodos.index(mes_destino_input)
            periodos_validos = lista_periodos[:idx_corte + 1]
            df = df[df['Periodo'].isin(periodos_validos)].copy()

            if not df.empty:
                ind_dest = indices_base[mes_destino_input]
                df['Venta_R'] = df.apply(
                    lambda row: row['Venta_H'] * (ind_dest / indices_base[row['Periodo']]) 
                    if pd.notnull(row['Venta_H']) and row['Periodo'] in indices_base else np.nan, 
                    axis=1
                )
                df['Fecha'] = pd.to_datetime(df['Periodo'], format='%Y/%m')
                
                def calcular_nombre_ejercicio(row):
                    if mes_inicio_num == 1: return f"{row.year}/{row.year}"
                    if row.month >= mes_inicio_num: return f"{row.year}/{row.year + 1}"
                    else: return f"{row.year - 1}/{row.year}"

                df['Ejercicio'] = df['Fecha'].apply(calcular_nombre_ejercicio)
                df['Mes_Etiqueta'] = df['Fecha'].apply(lambda row: f"{row.month:02d}. {meses_nombres[row.month-1]}")

                orden_meses = []
                for i in range(12):
                    m = (mes_inicio_num + i - 1) % 12
                    orden_meses.append(f"{m+1:02d}. {meses_nombres[m]}")

                # Creamos la matriz base solo con los años
                matriz = df.pivot_table(index='Mes_Etiqueta', columns='Ejercicio', values='Venta_R', 
                                        aggfunc=lambda x: x.sum(min_count=1), dropna=False)
                matriz = matriz.reindex(orden_meses)

                ejercicios_disponibles = sorted(matriz.columns)
                matriz_analisis = pd.DataFrame(index=matriz.index)

                # Construimos la tabla final intercalando Var%
                for i, ej_actual in enumerate(ejercicios_disponibles):
                    matriz_analisis[ej_actual] = matriz[ej_actual]
                    if i > 0:
                        ej_previo = ejercicios_disponibles[i-1]
                        nombre_var = "Var %" + (" " * i) 
                        matriz_analisis[nombre_var] = ((matriz[ej_actual] / matriz[ej_previo]) - 1) * 100

                # Calculamos Totales y Promedios sobre la matriz original de solo años
                totales_base = matriz.sum(axis=0, skipna=True)
                promedios_base = matriz.mean(axis=0, skipna=True)

                # Creamos las filas de TOTAL y PROMEDIO para la tabla final
                fila_total = []
                fila_promedio = []

                for i, col_name in enumerate(matriz_analisis.columns):
                    if "Var %" in col_name:
                        # Buscamos los ejercicios reales para comparar
                        idx_ej_actual = ejercicios_disponibles[i // 2] # El ejercicio actual
                        idx_ej_previo = ejercicios_disponibles[(i // 2) - 1] # El ejercicio anterior
                        
                        # Var % para Totales
                        v_act = totales_base[idx_ej_actual]
                        v_prev = totales_base[idx_ej_previo]
                        var_t = ((v_act / v_prev) - 1) * 100 if v_prev and v_prev != 0 else np.nan
                        fila_total.append(var_t)
                        
                        # Var % para Promedios
                        p_act = promedios_base[idx_ej_actual]
                        p_prev = promedios_base[idx_ej_previo]
                        var_p = ((p_act / p_prev) - 1) * 100 if p_prev and p_prev != 0 else np.nan
                        fila_promedio.append(var_p)
                    else:
                        # Es una columna de año, traemos el valor directo
                        fila_total.append(totales_base[col_name])
                        fila_promedio.append(promedios_base[col_name])

                matriz_analisis.loc['TOTAL EJERCICIO'] = fila_total
                matriz_analisis.loc['PROMEDIO MENSUAL'] = fila_promedio

                # Formatos de visualización
                def format_contable_pct(val):
                    if pd.isna(val) or val == 0: return "-"
                    v = int(round(val))
                    return f"({abs(v)}%)" if v < 0 else f"{v}%"

                def format_valor(x):
                    if pd.isna(x) or x == 0: return "S/D"
                    return f"$ {x:,.0f}"

                def color_variacion(val):
                    if pd.isna(val) or isinstance(val, str): return ''
                    if val < -0.001: return 'color: red; font-weight: bold'
                    if val > 0.001: return 'color: green; font-weight: bold'
                    return ''

                st.subheader(f"✅ Cuadro Comparativo (Moneda de Cierre: {mes_destino_input})")
                styler = matriz_analisis.style.apply(
                    lambda s: ['font-weight: bold; background-color: #e8f4f8' if s.name == 'TOTAL EJERCICIO' else '' for _ in s], axis=1
                ).apply(
                    lambda s: ['font-style: italic; background-color: #f9f9f9' if s.name == 'PROMEDIO MENSUAL' else '' for _2 in s], axis=1
                )
                columnas_var = [c for c in matriz_analisis.columns if "Var %" in c]
                styler = styler.map(color_variacion, subset=columnas_var)
                formatos_pantalla = {col: (format_contable_pct if "Var %" in col else format_valor) for col in matriz_analisis.columns}
                st.dataframe(styler.format(formatos_pantalla), use_container_width=True, height=550)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    matriz_analisis.to_excel(writer, sheet_name='Reporte_Ventas')
                st.download_button(label="📥 Descargar Reporte en Excel", data=output.getvalue(), 
                                   file_name=f"Analisis_Ventas_{mes_destino_input.replace('/','-')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error al procesar: {e}")