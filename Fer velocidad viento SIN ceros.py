import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np

def calcular_velocidad_promedio_total(nombre_archivo_csv):
    """
    Calcula la velocidad promedio del viento (U_barra) y su desviación estándar
    para todo el periodo a partir de los datos de 5 minutos, excluyendo valores cero.
    También calcula la densidad de potencia eólica promedio (P̄/A).
    Además, genera gráficas separadas por períodos específicos en ventanas independientes.

    Args:
        nombre_archivo_csv (str): La ruta del archivo CSV.
    
    Returns:
        tuple: (promedio, desviación estándar, densidad de potencia) o (None, None, None)
    """
    try:
        # 1. Cargar los datos
        # ====================
        df = pd.read_csv(nombre_archivo_csv)
        print("Datos cargados exitosamente.")

        # --- Nombres de las columnas ---
        columna_velocidad = 'Wind Speed (m/sec)' 
        columna_fecha = 'Fecha'  # Ajusta según tu CSV
        columna_hora = 'Hora'   # Ajusta según tu CSV
        
        # 2. Preprocesamiento
        # ===================
        # Asegurar que la columna de velocidad sea numérica
        df[columna_velocidad] = pd.to_numeric(df[columna_velocidad], errors='coerce')
        
        # Combinar fecha y hora en una sola columna datetime
        df['DateTime'] = pd.to_datetime(df[columna_fecha] + ' ' + df[columna_hora], errors='coerce')
        
        # Limpiar datos nulos
        df.dropna(subset=[columna_velocidad, 'DateTime'], inplace=True)
        
        # Ordenar por fecha y hora
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        print(f"\nTotal de mediciones cargadas: {len(df)}")
        
        # 3. FILTRAR VALORES CERO
        # ========================
        print(f"Mediciones con velocidad = 0: {(df[columna_velocidad] == 0).sum()}")
        
        # Crear DataFrame sin valores cero para cálculos estadísticos
        df_sin_ceros = df[df[columna_velocidad] > 0].copy()
        
        print(f"Mediciones válidas (velocidad > 0): {len(df_sin_ceros)}")
        print(f"Período completo de medición: {df_sin_ceros['DateTime'].min()} hasta {df_sin_ceros['DateTime'].max()}")
        
        # 4. Cálculo del Promedio Directo TOTAL (excluyendo ceros)
        # =========================================================
        velocidad_promedio_total = df_sin_ceros[columna_velocidad].mean()
        velocidad_promedio_con_ceros = df[columna_velocidad].mean()
        
        # 5. Cálculo de la Desviación Estándar (σ_U) - excluyendo ceros
        # ==============================================================
        # Fórmula: σ_U = sqrt( (1/(N-1)) * Σ(U_i - Ū)² )
        N = len(df_sin_ceros)
        desviacion_estandar = df_sin_ceros[columna_velocidad].std(ddof=1)  # ddof=1 usa N-1 en el denominador
        
        # Verificación manual de la fórmula (opcional, para demostración)
        suma_diferencias_cuadradas = ((df_sin_ceros[columna_velocidad] - velocidad_promedio_total) ** 2).sum()
        desviacion_manual = np.sqrt(suma_diferencias_cuadradas / (N - 1))
        
        # 6. Cálculo de la Densidad de Potencia Eólica Promedio (P̄/A)
        # =============================================================
        # Constantes
        rho = 1.225  # Densidad del aire a nivel del mar (kg/m³)
        
        # 6.1. Calcular velocidad promedio por hora
        df_sin_ceros['Hora_del_anio'] = df_sin_ceros['DateTime'].dt.hour + \
                                         (df_sin_ceros['DateTime'].dt.dayofyear - 1) * 24
        
        velocidades_horarias = df_sin_ceros.groupby('Hora_del_anio')[columna_velocidad].mean()
        
        # 6.2. Calcular el factor de energía Ke
        # Ke = (1/(N*Ū³)) * Σ(Ui³)
        N_horas = len(velocidades_horarias)
        suma_cubos = (velocidades_horarias ** 3).sum()
        Ke = suma_cubos / (N_horas * (velocidad_promedio_total ** 3))
        
        # 6.3. Calcular la densidad de potencia eólica promedio
        # P̄/A = (1/2) * ρ * Ū³ * Ke
        densidad_potencia = 0.5 * rho * (velocidad_promedio_total ** 3) * Ke
        
        print(f"\n--- ESTADÍSTICAS GENERALES (Excluyendo velocidad = 0) ---")
        print(f"Velocidad promedio (Ū): {velocidad_promedio_total:.3f} m/s")
        print(f"Desviación estándar (σ_U): {desviacion_estandar:.3f} m/s")
        print(f"Coeficiente de variación: {(desviacion_estandar/velocidad_promedio_total)*100:.2f}%")
        print(f"Verificación manual σ_U: {desviacion_manual:.3f} m/s")
        
        print(f"\n--- DENSIDAD DE POTENCIA EÓLICA ---")
        print(f"Número de horas analizadas: {N_horas}")
        print(f"Factor de energía (Ke): {Ke:.4f}")
        print(f"Densidad de potencia promedio (P̄/A): {densidad_potencia:.2f} W/m²")
        print(f"Densidad del aire (ρ): {rho} kg/m³")
        
        print(f"\n--- COMPARACIÓN ---")
        print(f"Promedio incluyendo ceros: {velocidad_promedio_con_ceros:.3f} m/s")
        print(f"Promedio excluyendo ceros: {velocidad_promedio_total:.3f} m/s")
        print(f"Diferencia: {velocidad_promedio_total - velocidad_promedio_con_ceros:.3f} m/s")
        
        # 7. Dividir datos en dos períodos (sin ceros)
        # =============================================
        # Período 1: 1 Nov 2024 - 31 Dic 2024
        periodo1 = df_sin_ceros[(df_sin_ceros['DateTime'] >= '2024-11-01') & 
                                (df_sin_ceros['DateTime'] <= '2024-12-31 23:59:59')]
        
        # Período 2: 1 Ene 2025 - 31 Oct 2025
        periodo2 = df_sin_ceros[(df_sin_ceros['DateTime'] >= '2025-01-01') & 
                                (df_sin_ceros['DateTime'] <= '2025-10-31 23:59:59')]
        
        print(f"\nPeríodo 1 (Nov-Dic 2024): {len(periodo1)} mediciones válidas")
        print(f"Período 2 (Ene-Oct 2025): {len(periodo2)} mediciones válidas")
        
        # 8. Crear GRÁFICA 1 en ventana separada: Noviembre - Diciembre 2024
        # ===================================================================
        if len(periodo1) > 0:
            velocidad_promedio_p1 = periodo1[columna_velocidad].mean()
            desviacion_p1 = periodo1[columna_velocidad].std(ddof=1)
            velocidad_maxima_p1 = periodo1[columna_velocidad].max()
            velocidad_minima_p1 = periodo1[columna_velocidad].min()
            indice_max_p1 = periodo1[columna_velocidad].idxmax()
            tiempo_max_p1 = periodo1.loc[indice_max_p1, 'DateTime']
            
            # Calcular densidad de potencia para el período 1
            periodo1_copy = periodo1.copy()
            periodo1_copy['Hora_del_periodo'] = periodo1_copy['DateTime'].dt.hour + \
                                                (periodo1_copy['DateTime'].dt.dayofyear - 
                                                 periodo1_copy['DateTime'].min().dayofyear) * 24
            velocidades_horarias_p1 = periodo1_copy.groupby('Hora_del_periodo')[columna_velocidad].mean()
            N_horas_p1 = len(velocidades_horarias_p1)
            suma_cubos_p1 = (velocidades_horarias_p1 ** 3).sum()
            Ke_p1 = suma_cubos_p1 / (N_horas_p1 * (velocidad_promedio_p1 ** 3))
            densidad_potencia_p1 = 0.5 * rho * (velocidad_promedio_p1 ** 3) * Ke_p1
            
            fig1 = plt.figure(figsize=(16, 7))
            ax1 = fig1.add_subplot(111)
            
            ax1.plot(periodo1['DateTime'], periodo1[columna_velocidad], 
                     linewidth=0.8, color='steelblue', alpha=0.7, label='Velocidad del viento')
            
            ax1.axhline(y=velocidad_promedio_p1, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {velocidad_promedio_p1:.2f} m/s')
            
            # Agregar bandas de desviación estándar
            ax1.axhline(y=velocidad_promedio_p1 + desviacion_p1, color='green', 
                        linestyle=':', linewidth=1.5, alpha=0.5, label=f'±1σ ({desviacion_p1:.2f} m/s)')
            ax1.axhline(y=max(0, velocidad_promedio_p1 - desviacion_p1), color='green', 
                        linestyle=':', linewidth=1.5, alpha=0.5)
            
            # Sombreado del área de ±1 desviación estándar
            ax1.fill_between(periodo1['DateTime'], 
                            max(0, velocidad_promedio_p1 - desviacion_p1), 
                            velocidad_promedio_p1 + desviacion_p1, 
                            color='green', alpha=0.1, label='Rango ±1σ')
            
            ax1.axhline(y=velocidad_promedio_total, color='orange', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {velocidad_promedio_total:.2f} m/s')
            
            ax1.scatter(tiempo_max_p1, velocidad_maxima_p1, color='red', s=200, 
                        zorder=5, label=f'Máx: {velocidad_maxima_p1:.2f} m/s', marker='*')
            
            # Agregar líneas verticales para separar meses
            meses_p1 = pd.date_range(start='2024-11-01', end='2025-01-01', freq='MS')
            for mes in meses_p1:
                ax1.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            # Configurar el formato del eje X
            ax1.xaxis.set_major_locator(MonthLocator())
            ax1.xaxis.set_major_formatter(DateFormatter('%B\n%Y'))
            
            ax1.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Velocidad del Viento (m/s)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Período 1: Noviembre - Diciembre 2024\n' + 
                         f'Ū = {velocidad_promedio_p1:.2f} m/s, σ_U = {desviacion_p1:.2f} m/s, ' +
                         f'P̄/A = {densidad_potencia_p1:.2f} W/m²\n(Excluyendo velocidad = 0)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(bottom=0)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
            plt.tight_layout()
            
            print(f"\n--- PERÍODO 1 (Nov-Dic 2024) ---")
            print(f"Velocidad promedio: {velocidad_promedio_p1:.2f} m/s")
            print(f"Desviación estándar: {desviacion_p1:.2f} m/s")
            print(f"Factor de energía (Ke): {Ke_p1:.4f}")
            print(f"Densidad de potencia (P̄/A): {densidad_potencia_p1:.2f} W/m²")
            print(f"Velocidad máxima: {velocidad_maxima_p1:.2f} m/s en {tiempo_max_p1}")
            print(f"Velocidad mínima (>0): {velocidad_minima_p1:.2f} m/s")
            print(f"Rango típico (Ū ± σ): {max(0, velocidad_promedio_p1-desviacion_p1):.2f} a {velocidad_promedio_p1+desviacion_p1:.2f} m/s")
        else:
            fig1 = plt.figure(figsize=(10, 6))
            ax1 = fig1.add_subplot(111)
            ax1.text(0.5, 0.5, 'No hay datos disponibles para este período', 
                     ha='center', va='center', fontsize=14, transform=ax1.transAxes)
            ax1.set_title('Período 1: Noviembre - Diciembre 2024', fontsize=14, fontweight='bold')
        
        # 9. Crear GRÁFICA 2 en ventana separada: Enero - Octubre 2025
        # =============================================================
        if len(periodo2) > 0:
            velocidad_promedio_p2 = periodo2[columna_velocidad].mean()
            desviacion_p2 = periodo2[columna_velocidad].std(ddof=1)
            velocidad_maxima_p2 = periodo2[columna_velocidad].max()
            velocidad_minima_p2 = periodo2[columna_velocidad].min()
            indice_max_p2 = periodo2[columna_velocidad].idxmax()
            tiempo_max_p2 = periodo2.loc[indice_max_p2, 'DateTime']
            
            # Calcular densidad de potencia para el período 2
            periodo2_copy = periodo2.copy()
            periodo2_copy['Hora_del_periodo'] = periodo2_copy['DateTime'].dt.hour + \
                                                (periodo2_copy['DateTime'].dt.dayofyear - 
                                                 periodo2_copy['DateTime'].min().dayofyear) * 24
            velocidades_horarias_p2 = periodo2_copy.groupby('Hora_del_periodo')[columna_velocidad].mean()
            N_horas_p2 = len(velocidades_horarias_p2)
            suma_cubos_p2 = (velocidades_horarias_p2 ** 3).sum()
            Ke_p2 = suma_cubos_p2 / (N_horas_p2 * (velocidad_promedio_p2 ** 3))
            densidad_potencia_p2 = 0.5 * rho * (velocidad_promedio_p2 ** 3) * Ke_p2
            
            fig2 = plt.figure(figsize=(16, 7))
            ax2 = fig2.add_subplot(111)
            
            ax2.plot(periodo2['DateTime'], periodo2[columna_velocidad], 
                     linewidth=0.8, color='steelblue', alpha=0.7, label='Velocidad del viento')
            
            ax2.axhline(y=velocidad_promedio_p2, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {velocidad_promedio_p2:.2f} m/s')
            
            # Agregar bandas de desviación estándar
            ax2.axhline(y=velocidad_promedio_p2 + desviacion_p2, color='green', 
                        linestyle=':', linewidth=1.5, alpha=0.5, label=f'±1σ ({desviacion_p2:.2f} m/s)')
            ax2.axhline(y=max(0, velocidad_promedio_p2 - desviacion_p2), color='green', 
                        linestyle=':', linewidth=1.5, alpha=0.5)
            
            # Sombreado del área de ±1 desviación estándar
            ax2.fill_between(periodo2['DateTime'], 
                            max(0, velocidad_promedio_p2 - desviacion_p2), 
                            velocidad_promedio_p2 + desviacion_p2, 
                            color='green', alpha=0.1, label='Rango ±1σ')
            
            ax2.axhline(y=velocidad_promedio_total, color='orange', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {velocidad_promedio_total:.2f} m/s')
            
            ax2.scatter(tiempo_max_p2, velocidad_maxima_p2, color='red', s=200, 
                        zorder=5, label=f'Máx: {velocidad_maxima_p2:.2f} m/s', marker='*')
            
            # Agregar líneas verticales para separar meses
            meses_p2 = pd.date_range(start='2025-01-01', end='2025-11-01', freq='MS')
            for mes in meses_p2:
                ax2.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            # Configurar el formato del eje X
            ax2.xaxis.set_major_locator(MonthLocator())
            ax2.xaxis.set_major_formatter(DateFormatter('%B'))
            
            ax2.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Velocidad del Viento (m/s)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Período 2: Enero - Octubre 2025\n' + 
                         f'Ū = {velocidad_promedio_p2:.2f} m/s, σ_U = {desviacion_p2:.2f} m/s, ' +
                         f'P̄/A = {densidad_potencia_p2:.2f} W/m²\n(Excluyendo velocidad = 0)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(bottom=0)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
            plt.tight_layout()
            
            print(f"\n--- PERÍODO 2 (Ene-Oct 2025) ---")
            print(f"Velocidad promedio: {velocidad_promedio_p2:.2f} m/s")
            print(f"Desviación estándar: {desviacion_p2:.2f} m/s")
            print(f"Factor de energía (Ke): {Ke_p2:.4f}")
            print(f"Densidad de potencia (P̄/A): {densidad_potencia_p2:.2f} W/m²")
            print(f"Velocidad máxima: {velocidad_maxima_p2:.2f} m/s en {tiempo_max_p2}")
            print(f"Velocidad mínima (>0): {velocidad_minima_p2:.2f} m/s")
            print(f"Rango típico (Ū ± σ): {max(0, velocidad_promedio_p2-desviacion_p2):.2f} a {velocidad_promedio_p2+desviacion_p2:.2f} m/s")
        else:
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            ax2.text(0.5, 0.5, 'No hay datos disponibles para este período', 
                     ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Período 2: Enero - Octubre 2025', fontsize=14, fontweight='bold')
        
        # Mostrar todas las gráficas
        plt.show()
        
        # 10. Análisis mensual (excluyendo ceros)
        # ========================================
        print("\n--- VELOCIDAD PROMEDIO, DESVIACIÓN ESTÁNDAR Y DENSIDAD DE POTENCIA POR MES ---")
        print("(Excluyendo velocidad = 0)")
        df_sin_ceros['Mes'] = df_sin_ceros['DateTime'].dt.to_period('M')
        estadisticas_mensuales = df_sin_ceros.groupby('Mes')[columna_velocidad].agg(['mean', 'std', 'count'])
        
        # Calcular densidad de potencia por mes
        densidades_mensuales = []
        for mes in estadisticas_mensuales.index:
            datos_mes = df_sin_ceros[df_sin_ceros['Mes'] == mes]
            datos_mes_copy = datos_mes.copy()
            datos_mes_copy['Hora_mes'] = datos_mes_copy['DateTime'].dt.hour + \
                                          (datos_mes_copy['DateTime'].dt.day - 1) * 24
            vel_horarias_mes = datos_mes_copy.groupby('Hora_mes')[columna_velocidad].mean()
            N_h = len(vel_horarias_mes)
            vel_prom_mes = estadisticas_mensuales.loc[mes, 'mean']
            if N_h > 0 and vel_prom_mes > 0:
                Ke_mes = (vel_horarias_mes ** 3).sum() / (N_h * (vel_prom_mes ** 3))
                dens_pot_mes = 0.5 * rho * (vel_prom_mes ** 3) * Ke_mes
            else:
                dens_pot_mes = 0
            densidades_mensuales.append(dens_pot_mes)
        
        estadisticas_mensuales['Densidad Potencia (W/m²)'] = densidades_mensuales
        estadisticas_mensuales = estadisticas_mensuales.sort_values('mean', ascending=False)
        
        print(estadisticas_mensuales.to_string())
        
        mes_mas_ventoso = estadisticas_mensuales['mean'].idxmax()
        print(f"\nMes más ventoso: {mes_mas_ventoso}")
        print(f"  - Velocidad promedio: {estadisticas_mensuales.loc[mes_mas_ventoso, 'mean']:.2f} m/s")
        print(f"  - Densidad de potencia: {estadisticas_mensuales.loc[mes_mas_ventoso, 'Densidad Potencia (W/m²)']:.2f} W/m²")
        
        # 11. Top 10 velocidades más altas (general)
        # ===========================================
        print("\n--- TOP 10 VELOCIDADES MÁS ALTAS (TODO EL PERÍODO) ---")
        top_10 = df_sin_ceros.nlargest(10, columna_velocidad)[['DateTime', columna_velocidad]]
        top_10_display = top_10.copy()
        top_10_display['DateTime'] = top_10_display['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        print(top_10_display.to_string(index=False))
        
        return velocidad_promedio_total, desviacion_estandar, densidad_potencia

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{nombre_archivo_csv}'.")
        return None, None, None
    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# --- Ejecución del código ---
archivo = r'C:\Users\JuanF\Downloads\datos_simplificados.csv'
promedio_final, desviacion_final, densidad_pot_final = calcular_velocidad_promedio_total(archivo)

if promedio_final is not None:
    print("\n--- RESULTADOS FINALES ---")
    print(f"Velocidad promedio del viento (Ū): {promedio_final:.2f} m/s")
    print(f"Desviación estándar (σ_U): {desviacion_final:.2f} m/s")
    print(f"Densidad de potencia eólica promedio (P̄/A): {densidad_pot_final:.2f} W/m²")
    print(f"Intervalo de confianza (68%): {max(0, promedio_final-desviacion_final):.2f} a {promedio_final+desviacion_final:.2f} m/s")
    print("(Calculado excluyendo mediciones con velocidad = 0)")
