import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy.interpolate import make_interp_spline

def calcular_promedio_irradiacion_acumulada(nombre_archivo_csv):
    """
    Calcula el promedio de la Irradiación Solar Acumulada Diaria para todo el período
    (en kWh/m²/día) a partir de datos de 5 minutos, incluyendo valores cero.
    También calcula la irradiancia promedio diaria (W/m²) y su desviación estándar.
    Además, genera gráficas separadas por períodos específicos en ventanas independientes.
    """
    try:
        df = pd.read_csv(nombre_archivo_csv)
        print("Datos cargados exitosamente.")

        # --- Nombre de la columna de radiación ---
        columna_radiacion = 'Solar Radiation (W/m^2)' 
        
        # 1. Preprocesamiento
        df['Timestamp'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
        df.set_index('Timestamp', inplace=True)
        
        # Usar el nombre de columna correcto y limpiar nulos
        df[columna_radiacion] = pd.to_numeric(df[columna_radiacion], errors='coerce')
        df.dropna(subset=[columna_radiacion], inplace=True)
        
        # 2. Convertir Potencia (W/m²) a Energía (kWh/m²) para cada intervalo
        tiempo_en_horas = 5 / 60
        df['Energia_kWh_m2'] = df[columna_radiacion] * tiempo_en_horas / 1000
        
        # 2.5. Cálculo de la Irradiancia Promedio por Día (W/m²)
        # =======================================================
        # Para cada día se hace el promedio de la columna original de irradiancia instantánea
        irradiancia_promedio_diaria = df[columna_radiacion].resample('D').mean()
        
        print("\n--- IRRADIANCIA PROMEDIO DIARIA (W/m²) ---")
        print(f"Total de días con datos: {len(irradiancia_promedio_diaria)}")
        
        # Calcular el promedio general del período
        promedio_irradiancia_periodo = irradiancia_promedio_diaria.mean()
        
        # Calcular la desviación estándar de la irradiancia diaria
        desviacion_std_irradiancia = irradiancia_promedio_diaria.std()
        
        print(f"\n--- ESTADÍSTICAS DE IRRADIANCIA (W/m²) ---")
        print(f"Promedio de irradiancia diaria: {promedio_irradiancia_periodo:.2f} W/m²")
        print(f"Desviación estándar: {desviacion_std_irradiancia:.2f} W/m²")
        
        # Mostrar primeros 10 días como ejemplo
        print("\n--- PRIMEROS 10 DÍAS (Irradiancia Promedio Diaria) ---")
        for fecha, valor in irradiancia_promedio_diaria.head(10).items():
            print(f"{fecha.date()}: {valor:.2f} W/m²")
        
        # 3. Sumar la Energía Acumulada por Día (Irradiación Diaria)
        irradiacion_diaria = df['Energia_kWh_m2'].resample('D').sum()
        
        # 4. Información sobre los datos
        print(f"\n--- IRRADIACIÓN ACUMULADA DIARIA (kWh/m²/día) ---")
        print(f"Total de días calculados: {len(irradiacion_diaria)}")
        print(f"Días con irradiación cero: {(irradiacion_diaria == 0).sum()}")
        print(f"Período completo: {irradiacion_diaria.index.min()} hasta {irradiacion_diaria.index.max()}")
        
        # 5. Calcular el Promedio General (incluyendo ceros)
        promedio_acumulado_final = irradiacion_diaria.mean()
        
        print(f"\n--- PROMEDIO DE IRRADIACIÓN (kWh/m²/día) ---")
        print(f"Promedio de irradiación diaria: {promedio_acumulado_final:.3f} kWh/m²/día")
        
        # 6. Dividir datos en dos períodos
        # =================================
        # Período 1: 1 Nov 2024 - 31 Dic 2024
        periodo1 = irradiacion_diaria[(irradiacion_diaria.index >= '2024-11-01') & 
                                      (irradiacion_diaria.index <= '2024-12-31')]
        
        periodo1_irradiancia = irradiancia_promedio_diaria[
            (irradiancia_promedio_diaria.index >= '2024-11-01') & 
            (irradiancia_promedio_diaria.index <= '2024-12-31')]
        
        # Período 2: 1 Ene 2025 - 31 Oct 2025
        periodo2 = irradiacion_diaria[(irradiacion_diaria.index >= '2025-01-01') & 
                                      (irradiacion_diaria.index <= '2025-10-31')]
        
        periodo2_irradiancia = irradiancia_promedio_diaria[
            (irradiancia_promedio_diaria.index >= '2025-01-01') & 
            (irradiancia_promedio_diaria.index <= '2025-10-31')]
        
        print(f"\nPeríodo 1 (Nov-Dic 2024): {len(periodo1)} días")
        print(f"Período 2 (Ene-Oct 2025): {len(periodo2)} días")
        
        # 7. Crear GRÁFICA 1 en ventana separada: Noviembre - Diciembre 2024
        # ===================================================================
        if len(periodo1) > 0:
            promedio_p1 = periodo1.mean()
            promedio_irradiancia_p1 = periodo1_irradiancia.mean()
            std_irradiancia_p1 = periodo1_irradiancia.std()
            maximo_p1 = periodo1.max()
            fecha_max_p1 = periodo1.idxmax()
            minimo_p1 = periodo1.min()
            
            fig1 = plt.figure(figsize=(16, 7))
            ax1 = fig1.add_subplot(111)
            
            # Datos originales con transparencia
            ax1.scatter(periodo1.index, periodo1.values, 
                       color='orange', alpha=0.3, s=20, label='Datos diarios', zorder=2)
            
            # Crear curva suave usando spline
            x_numeric = np.arange(len(periodo1))
            if len(periodo1) > 3:  # Necesitamos al menos 4 puntos para spline
                spline = make_interp_spline(x_numeric, periodo1.values, k=3)
                x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 300)
                y_smooth = spline(x_smooth)
                # Convertir índices numéricos de vuelta a fechas
                fechas_smooth = pd.date_range(start=periodo1.index[0], 
                                             end=periodo1.index[-1], 
                                             periods=300)
                ax1.plot(fechas_smooth, y_smooth, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia suavizada', zorder=3)
            else:
                ax1.plot(periodo1.index, periodo1.values, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia', zorder=3)
            
            ax1.axhline(y=promedio_p1, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {promedio_p1:.3f} kWh/m²/día')
            
            ax1.axhline(y=promedio_acumulado_final, color='blue', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {promedio_acumulado_final:.3f} kWh/m²/día')
            
            ax1.scatter(fecha_max_p1, maximo_p1, color='red', s=300, 
                        zorder=5, label=f'Máx: {maximo_p1:.3f} kWh/m²/día', marker='*', 
                        edgecolors='darkred', linewidth=1.5)
            
            # Agregar líneas verticales para separar meses
            meses_p1 = pd.date_range(start='2024-11-01', end='2025-01-01', freq='MS')
            for mes in meses_p1:
                ax1.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            # Configurar el formato del eje X
            ax1.xaxis.set_major_locator(MonthLocator())
            ax1.xaxis.set_major_formatter(DateFormatter('%B\n%Y'))
            
            ax1.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Irradiación Solar (kWh/m²/día)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Período 1: Irradiación Solar Diaria - Noviembre-Diciembre 2024\n' + 
                         f'Irradiación: {promedio_p1:.3f} kWh/m²/día | Irradiancia: {promedio_irradiancia_p1:.2f} ± {std_irradiancia_p1:.2f} W/m²', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_ylim(bottom=0)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
            plt.tight_layout()
            
            print(f"\n--- PERÍODO 1 (Nov-Dic 2024) ---")
            print(f"Irradiación promedio: {promedio_p1:.3f} kWh/m²/día")
            print(f"Irradiancia promedio: {promedio_irradiancia_p1:.2f} W/m²")
            print(f"Desviación estándar irradiancia: {std_irradiancia_p1:.2f} W/m²")
            print(f"Irradiación máxima: {maximo_p1:.3f} kWh/m²/día en {fecha_max_p1.date()}")
            print(f"Irradiación mínima: {minimo_p1:.3f} kWh/m²/día")
        else:
            fig1 = plt.figure(figsize=(10, 6))
            ax1 = fig1.add_subplot(111)
            ax1.text(0.5, 0.5, 'No hay datos disponibles para este período', 
                     ha='center', va='center', fontsize=14, transform=ax1.transAxes)
            ax1.set_title('Período 1: Noviembre - Diciembre 2024', fontsize=14, fontweight='bold')
        
        # 8. Crear GRÁFICA 2 en ventana separada: Enero - Octubre 2025
        # =============================================================
        if len(periodo2) > 0:
            promedio_p2 = periodo2.mean()
            promedio_irradiancia_p2 = periodo2_irradiancia.mean()
            std_irradiancia_p2 = periodo2_irradiancia.std()
            maximo_p2 = periodo2.max()
            fecha_max_p2 = periodo2.idxmax()
            minimo_p2 = periodo2.min()
            
            fig2 = plt.figure(figsize=(16, 7))
            ax2 = fig2.add_subplot(111)
            
            # Datos originales con transparencia
            ax2.scatter(periodo2.index, periodo2.values, 
                       color='orange', alpha=0.3, s=20, label='Datos diarios', zorder=2)
            
            # Crear curva suave usando spline
            x_numeric = np.arange(len(periodo2))
            if len(periodo2) > 3:
                spline = make_interp_spline(x_numeric, periodo2.values, k=3)
                x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 500)
                y_smooth = spline(x_smooth)
                # Convertir índices numéricos de vuelta a fechas
                fechas_smooth = pd.date_range(start=periodo2.index[0], 
                                             end=periodo2.index[-1], 
                                             periods=500)
                ax2.plot(fechas_smooth, y_smooth, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia suavizada', zorder=3)
            else:
                ax2.plot(periodo2.index, periodo2.values, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia', zorder=3)
            
            ax2.axhline(y=promedio_p2, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {promedio_p2:.3f} kWh/m²/día')
            
            ax2.axhline(y=promedio_acumulado_final, color='blue', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {promedio_acumulado_final:.3f} kWh/m²/día')
            
            ax2.scatter(fecha_max_p2, maximo_p2, color='red', s=300, 
                        zorder=5, label=f'Máx: {maximo_p2:.3f} kWh/m²/día', marker='*',
                        edgecolors='darkred', linewidth=1.5)
            
            # Agregar líneas verticales para separar meses
            meses_p2 = pd.date_range(start='2025-01-01', end='2025-11-01', freq='MS')
            for mes in meses_p2:
                ax2.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            # Configurar el formato del eje X
            ax2.xaxis.set_major_locator(MonthLocator())
            ax2.xaxis.set_major_formatter(DateFormatter('%B'))
            
            ax2.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Irradiación Solar (kWh/m²/día)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Período 2: Irradiación Solar Diaria - Enero-Octubre 2025\n' + 
                         f'Irradiación: {promedio_p2:.3f} kWh/m²/día | Irradiancia: {promedio_irradiancia_p2:.2f} ± {std_irradiancia_p2:.2f} W/m²', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(bottom=0)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
            plt.tight_layout()
            
            print(f"\n--- PERÍODO 2 (Ene-Oct 2025) ---")
            print(f"Irradiación promedio: {promedio_p2:.3f} kWh/m²/día")
            print(f"Irradiancia promedio: {promedio_irradiancia_p2:.2f} W/m²")
            print(f"Desviación estándar irradiancia: {std_irradiancia_p2:.2f} W/m²")
            print(f"Irradiación máxima: {maximo_p2:.3f} kWh/m²/día en {fecha_max_p2.date()}")
            print(f"Irradiación mínima: {minimo_p2:.3f} kWh/m²/día")
        else:
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            ax2.text(0.5, 0.5, 'No hay datos disponibles para este período', 
                     ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Período 2: Enero - Octubre 2025', fontsize=14, fontweight='bold')
        
        # Mostrar todas las gráficas
        plt.show()
        
        # 9. Análisis mensual
        # ====================
        print("\n--- IRRADIACIÓN E IRRADIANCIA PROMEDIO POR MES ---")
        
        # Calcular promedio mensual de irradiación (kWh/m²/día)
        irradiacion_mensual = irradiacion_diaria.resample('M').mean()
        
        # Calcular promedio mensual de irradiancia (W/m²)
        irradiancia_mensual = irradiancia_promedio_diaria.resample('M').mean()
        
        # Calcular desviación estándar mensual de irradiancia
        irradiancia_std_mensual = irradiancia_promedio_diaria.resample('M').std()
        
        # Combinar en un DataFrame
        estadisticas_mensuales = pd.DataFrame({
            'Irradiación (kWh/m²/día)': irradiacion_mensual,
            'Irradiancia (W/m²)': irradiancia_mensual,
            'Desv. Std. Irradiancia (W/m²)': irradiancia_std_mensual
        })
        
        estadisticas_mensuales_sorted = estadisticas_mensuales.sort_values('Irradiación (kWh/m²/día)', ascending=False)
        
        for fecha, row in estadisticas_mensuales_sorted.iterrows():
            # Contar cuántos días tuvo ese mes
            dias_mes = irradiacion_diaria[irradiacion_diaria.index.to_period('M') == fecha.to_period('M')]
            print(f"{fecha.strftime('%Y-%m')}: {row['Irradiación (kWh/m²/día)']:.3f} kWh/m²/día | "
                  f"{row['Irradiancia (W/m²)']:.2f} ± {row['Desv. Std. Irradiancia (W/m²)']:.2f} W/m² ({len(dias_mes)} días)")
        
        mes_mas_soleado = estadisticas_mensuales['Irradiación (kWh/m²/día)'].idxmax()
        print(f"\nMes más soleado: {mes_mas_soleado.strftime('%B %Y')} con:")
        print(f"  - Irradiación: {estadisticas_mensuales.loc[mes_mas_soleado, 'Irradiación (kWh/m²/día)']:.3f} kWh/m²/día")
        print(f"  - Irradiancia: {estadisticas_mensuales.loc[mes_mas_soleado, 'Irradiancia (W/m²)']:.2f} ± "
              f"{estadisticas_mensuales.loc[mes_mas_soleado, 'Desv. Std. Irradiancia (W/m²)']:.2f} W/m²")
        
        # 10. Top 10 días con mayor irradiación
        # ======================================
        print("\n--- TOP 10 DÍAS CON MAYOR IRRADIACIÓN ---")
        top_10 = irradiacion_diaria.nlargest(10)
        for fecha, valor in top_10.items():
            # Obtener la irradiancia promedio de ese día
            irradiancia_dia = irradiancia_promedio_diaria[fecha] if fecha in irradiancia_promedio_diaria.index else 0
            print(f"{fecha.date()}: {valor:.3f} kWh/m²/día ({irradiancia_dia:.2f} W/m²)")
        
        return promedio_acumulado_final, promedio_irradiancia_periodo, desviacion_std_irradiancia

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
promedio_acumulado, promedio_irradiancia, std_irradiancia = calcular_promedio_irradiacion_acumulada(archivo)

if promedio_acumulado is not None:
    print("\n--- RESULTADOS FINALES ---")
    print(f"Irradiación promedio diaria (GHI): {promedio_acumulado:.3f} kWh/m²/día")
    print(f"Irradiancia promedio diaria: {promedio_irradiancia:.2f} ± {std_irradiancia:.2f} W/m²")