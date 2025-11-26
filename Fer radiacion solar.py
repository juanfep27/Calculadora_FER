import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy.interpolate import make_interp_spline

def analisis_completo_irradiacion(nombre_archivo_csv):
    """
    Análisis completo de irradiación solar que incluye:
    1. Curva de irradiancia promedio por hora del día
    2. Análisis temporal con gráficas por períodos
    3. Estadísticas mensuales y diarias
    """
    try:
        # ==================== CARGA Y PREPROCESAMIENTO ====================
        df = pd.read_csv(nombre_archivo_csv)
        print("="*70)
        print("DATOS CARGADOS EXITOSAMENTE")
        print("="*70)
        
        columna_radiacion = 'Solar Radiation (W/m^2)'
        
        # Preprocesamiento común
        df['Timestamp'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
        df[columna_radiacion] = pd.to_numeric(df[columna_radiacion], errors='coerce')
        df.dropna(subset=[columna_radiacion], inplace=True)
        
        # ==================== PARTE 1: CURVA HORARIA ====================
        print("\n" + "="*70)
        print("PARTE 1: ANÁLISIS DE IRRADIANCIA POR HORA DEL DÍA")
        print("="*70)
        
        # Extraer la hora del día
        df['Hora_del_dia'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60
        
        # Calcular promedio de irradiancia por hora del día
        irradiancia_por_hora = df.groupby('Hora_del_dia')[columna_radiacion].mean()
        
        # Estadísticas horarias
        irradiancia_maxima = irradiancia_por_hora.max()
        hora_maxima = irradiancia_por_hora.idxmax()
        irradiancia_promedio_horario = irradiancia_por_hora.mean()
        
        print(f"\nEstadísticas de la curva diaria:")
        print(f"  • Irradiancia máxima: {irradiancia_maxima:.2f} W/m² a las {hora_maxima:.2f} horas")
        print(f"  • Irradiancia promedio del día: {irradiancia_promedio_horario:.2f} W/m²")
        
        # GRÁFICA 1: Curva horaria
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        ax1.set_facecolor('#FFF8DC')
        
        # Área bajo la curva
        ax1.fill_between(irradiancia_por_hora.index, 0, irradiancia_por_hora.values,
                        color='#FFD700', alpha=0.7, label='Irradiación Solar Acumulada')
        
        # Curva de irradiancia
        ax1.plot(irradiancia_por_hora.index, irradiancia_por_hora.values,
               color='#FF8C00', linewidth=3, label='Curva de Irradiancia')
        
        # Línea promedio
        ax1.axhline(y=irradiancia_promedio_horario, color='red', linestyle='--', 
                  linewidth=2, label=f'Irradiancia Promedio: {irradiancia_promedio_horario:.0f} W/m²')
        
        # Punto máximo
        ax1.plot(hora_maxima, irradiancia_maxima, 'o', 
               color='red', markersize=12, zorder=5,
               label=f'Máximo: {irradiancia_maxima:.0f} W/m² ({hora_maxima:.1f}h)')
        
        # Anotaciones
        hora_amanecer = irradiancia_por_hora[irradiancia_por_hora > 10].index.min()
        hora_atardecer = irradiancia_por_hora[irradiancia_por_hora > 10].index.max()
        
        if not pd.isna(hora_amanecer):
            ax1.annotate('AMANECER', xy=(hora_amanecer, 50), 
                       xytext=(hora_amanecer - 1, irradiancia_maxima * 0.3),
                       fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        ax1.annotate('MEDIODÍA', xy=(hora_maxima, irradiancia_maxima), 
                   xytext=(hora_maxima, irradiancia_maxima * 1.15),
                   fontsize=11, fontweight='bold', ha='center',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        if not pd.isna(hora_atardecer):
            ax1.annotate('ATARDECER', xy=(hora_atardecer, 50), 
                       xytext=(hora_atardecer + 0.5, irradiancia_maxima * 0.3),
                       fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Configuración
        ax1.set_xlabel('HORA DEL DÍA', fontsize=13, fontweight='bold')
        ax1.set_ylabel('IRRADIANCIA (W/m²)', fontsize=13, fontweight='bold')
        ax1.set_title('CURVA DE IRRADIANCIA SOLAR DIARIA PROMEDIO\n' + 
                    'Irradiación = Área bajo la curva de irradiancia',
                    fontsize=15, fontweight='bold', pad=20)
        
        horas_labels = range(0, 25, 2)
        ax1.set_xticks(horas_labels)
        ax1.set_xticklabels([f'{h:02d}:00' for h in horas_labels], rotation=45)
        ax1.set_xlim(0, 24)
        ax1.set_ylim(0, irradiancia_maxima * 1.2)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        nota = ('Nota: La irradiación diaria total (kWh/m²/día) es igual al área\n'
                'bajo la curva de irradiancia instantánea (W/m²)')
        fig1.text(0.99, 0.01, nota, ha='right', va='bottom', 
                fontsize=9, style='italic', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # ==================== PARTE 2: ANÁLISIS TEMPORAL ====================
        print("\n" + "="*70)
        print("PARTE 2: ANÁLISIS TEMPORAL DE IRRADIACIÓN")
        print("="*70)
        
        # Establecer índice temporal
        df_temporal = df.set_index('Timestamp')
        
        # Convertir Potencia a Energía
        tiempo_en_horas = 5 / 60
        df_temporal['Energia_kWh_m2'] = df_temporal[columna_radiacion] * tiempo_en_horas / 1000
        
        # Irradiancia promedio diaria
        irradiancia_promedio_diaria = df_temporal[columna_radiacion].resample('D').mean()
        irradiancia_promedio_diaria_sin_ceros = irradiancia_promedio_diaria[irradiancia_promedio_diaria > 0]
        
        # Irradiación acumulada diaria
        irradiacion_diaria = df_temporal['Energia_kWh_m2'].resample('D').sum()
        irradiacion_diaria_sin_ceros = irradiacion_diaria[irradiacion_diaria > 0]
        
        print(f"\nDatos procesados:")
        print(f"  • Total de días: {len(irradiacion_diaria)}")
        print(f"  • Días con irradiación > 0: {len(irradiacion_diaria_sin_ceros)}")
        print(f"  • Días excluidos (irradiación = 0): {len(irradiacion_diaria) - len(irradiacion_diaria_sin_ceros)}")
        print(f"  • Período: {irradiacion_diaria.index.min().date()} hasta {irradiacion_diaria.index.max().date()}")
        
        # Promedios generales
        promedio_irradiacion = irradiacion_diaria_sin_ceros.mean()
        promedio_irradiancia = irradiancia_promedio_diaria_sin_ceros.mean()
        std_irradiancia = irradiancia_promedio_diaria_sin_ceros.std()
        
        print(f"\nPromedios del período completo (sin ceros):")
        print(f"  • Irradiación diaria: {promedio_irradiacion:.3f} kWh/m²/día")
        print(f"  • Irradiancia diaria: {promedio_irradiancia:.2f} ± {std_irradiancia:.2f} W/m²")
        
        # Dividir en períodos
        periodo1 = irradiacion_diaria_sin_ceros[(irradiacion_diaria_sin_ceros.index >= '2024-11-01') & 
                                                (irradiacion_diaria_sin_ceros.index <= '2024-12-31')]
        periodo1_irradiancia = irradiancia_promedio_diaria_sin_ceros[
            (irradiancia_promedio_diaria_sin_ceros.index >= '2024-11-01') & 
            (irradiancia_promedio_diaria_sin_ceros.index <= '2024-12-31')]
        
        periodo2 = irradiacion_diaria_sin_ceros[(irradiacion_diaria_sin_ceros.index >= '2025-01-01') & 
                                                (irradiacion_diaria_sin_ceros.index <= '2025-10-31')]
        periodo2_irradiancia = irradiancia_promedio_diaria_sin_ceros[
            (irradiancia_promedio_diaria_sin_ceros.index >= '2025-01-01') & 
            (irradiancia_promedio_diaria_sin_ceros.index <= '2025-10-31')]
        
        # GRÁFICA 2: Período 1 (Nov-Dic 2024)
        if len(periodo1) > 0:
            promedio_p1 = periodo1.mean()
            promedio_irr_p1 = periodo1_irradiancia.mean()
            std_irr_p1 = periodo1_irradiancia.std()
            maximo_p1 = periodo1.max()
            fecha_max_p1 = periodo1.idxmax()
            
            fig2 = plt.figure(figsize=(16, 7))
            ax2 = fig2.add_subplot(111)
            
            ax2.scatter(periodo1.index, periodo1.values, 
                       color='orange', alpha=0.3, s=20, label='Datos diarios', zorder=2)
            
            x_numeric = np.arange(len(periodo1))
            if len(periodo1) > 3:
                spline = make_interp_spline(x_numeric, periodo1.values, k=3)
                x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 300)
                y_smooth = spline(x_smooth)
                fechas_smooth = pd.date_range(start=periodo1.index[0], 
                                             end=periodo1.index[-1], 
                                             periods=300)
                ax2.plot(fechas_smooth, y_smooth, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia suavizada', zorder=3)
            
            ax2.axhline(y=promedio_p1, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {promedio_p1:.3f} kWh/m²/día')
            ax2.axhline(y=promedio_irradiacion, color='blue', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {promedio_irradiacion:.3f} kWh/m²/día')
            ax2.scatter(fecha_max_p1, maximo_p1, color='red', s=300, 
                        zorder=5, label=f'Máx: {maximo_p1:.3f} kWh/m²/día', marker='*', 
                        edgecolors='darkred', linewidth=1.5)
            
            meses_p1 = pd.date_range(start='2024-11-01', end='2025-01-01', freq='MS')
            for mes in meses_p1:
                ax2.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            ax2.xaxis.set_major_locator(MonthLocator())
            ax2.xaxis.set_major_formatter(DateFormatter('%B\n%Y'))
            ax2.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Irradiación Solar (kWh/m²/día)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Período 1: Irradiación Solar Diaria - Noviembre-Diciembre 2024 (Sin Ceros)\n' + 
                         f'Irradiación: {promedio_p1:.3f} kWh/m²/día | Irradiancia: {promedio_irr_p1:.2f} ± {std_irr_p1:.2f} W/m²', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(bottom=0)
            plt.tight_layout()
            
            print(f"\nPeríodo 1 (Nov-Dic 2024):")
            print(f"  • Días con datos: {len(periodo1)}")
            print(f"  • Irradiación promedio: {promedio_p1:.3f} kWh/m²/día")
            print(f"  • Irradiancia promedio: {promedio_irr_p1:.2f} ± {std_irr_p1:.2f} W/m²")
            print(f"  • Máxima irradiación: {maximo_p1:.3f} kWh/m²/día ({fecha_max_p1.date()})")
        
        # GRÁFICA 3: Período 2 (Ene-Oct 2025)
        if len(periodo2) > 0:
            promedio_p2 = periodo2.mean()
            promedio_irr_p2 = periodo2_irradiancia.mean()
            std_irr_p2 = periodo2_irradiancia.std()
            maximo_p2 = periodo2.max()
            fecha_max_p2 = periodo2.idxmax()
            
            fig3 = plt.figure(figsize=(16, 7))
            ax3 = fig3.add_subplot(111)
            
            ax3.scatter(periodo2.index, periodo2.values, 
                       color='orange', alpha=0.3, s=20, label='Datos diarios', zorder=2)
            
            x_numeric = np.arange(len(periodo2))
            if len(periodo2) > 3:
                spline = make_interp_spline(x_numeric, periodo2.values, k=3)
                x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 500)
                y_smooth = spline(x_smooth)
                fechas_smooth = pd.date_range(start=periodo2.index[0], 
                                             end=periodo2.index[-1], 
                                             periods=500)
                ax3.plot(fechas_smooth, y_smooth, 
                        linewidth=2.5, color='darkorange', alpha=0.9, 
                        label='Tendencia suavizada', zorder=3)
            
            ax3.axhline(y=promedio_p2, color='green', 
                        linestyle='--', linewidth=2, label=f'Promedio período: {promedio_p2:.3f} kWh/m²/día')
            ax3.axhline(y=promedio_irradiacion, color='blue', 
                        linestyle=':', linewidth=2, alpha=0.7, label=f'Promedio total: {promedio_irradiacion:.3f} kWh/m²/día')
            ax3.scatter(fecha_max_p2, maximo_p2, color='red', s=300, 
                        zorder=5, label=f'Máx: {maximo_p2:.3f} kWh/m²/día', marker='*',
                        edgecolors='darkred', linewidth=1.5)
            
            meses_p2 = pd.date_range(start='2025-01-01', end='2025-11-01', freq='MS')
            for mes in meses_p2:
                ax3.axvline(x=mes, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
            
            ax3.xaxis.set_major_locator(MonthLocator())
            ax3.xaxis.set_major_formatter(DateFormatter('%B'))
            ax3.set_xlabel('Mes', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Irradiación Solar (kWh/m²/día)', fontsize=12, fontweight='bold')
            ax3.set_title(f'Período 2: Irradiación Solar Diaria - Enero-Octubre 2025 (Sin Ceros)\n' + 
                         f'Irradiación: {promedio_p2:.3f} kWh/m²/día | Irradiancia: {promedio_irr_p2:.2f} ± {std_irr_p2:.2f} W/m²', 
                         fontsize=14, fontweight='bold', pad=20)
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.set_ylim(bottom=0)
            plt.tight_layout()
            
            print(f"\nPeríodo 2 (Ene-Oct 2025):")
            print(f"  • Días con datos: {len(periodo2)}")
            print(f"  • Irradiación promedio: {promedio_p2:.3f} kWh/m²/día")
            print(f"  • Irradiancia promedio: {promedio_irr_p2:.2f} ± {std_irr_p2:.2f} W/m²")
            print(f"  • Máxima irradiación: {maximo_p2:.3f} kWh/m²/día ({fecha_max_p2.date()})")
        
        # ==================== ANÁLISIS MENSUAL ====================
        print("\n" + "="*70)
        print("ANÁLISIS MENSUAL (SIN CEROS)")
        print("="*70)
        
        def promedio_mensual_sin_ceros(serie):
            resultado = {}
            for mes_fecha, mes_datos in serie.resample('M'):
                datos_sin_ceros = mes_datos[mes_datos > 0]
                if len(datos_sin_ceros) > 0:
                    resultado[mes_fecha] = datos_sin_ceros.mean()
            return pd.Series(resultado)
        
        def std_mensual_sin_ceros(serie):
            resultado = {}
            for mes_fecha, mes_datos in serie.resample('M'):
                datos_sin_ceros = mes_datos[mes_datos > 0]
                if len(datos_sin_ceros) > 0:
                    resultado[mes_fecha] = datos_sin_ceros.std()
            return pd.Series(resultado)
        
        irradiacion_mensual = promedio_mensual_sin_ceros(irradiacion_diaria_sin_ceros)
        irradiancia_mensual = promedio_mensual_sin_ceros(irradiancia_promedio_diaria_sin_ceros)
        irradiancia_std_mensual = std_mensual_sin_ceros(irradiancia_promedio_diaria_sin_ceros)
        
        estadisticas_mensuales = pd.DataFrame({
            'Irradiación (kWh/m²/día)': irradiacion_mensual,
            'Irradiancia (W/m²)': irradiancia_mensual,
            'Desv. Std. Irradiancia (W/m²)': irradiancia_std_mensual
        }).dropna().sort_values('Irradiación (kWh/m²/día)', ascending=False)
        
        print("\nEstadísticas mensuales ordenadas por irradiación:")
        for fecha, row in estadisticas_mensuales.iterrows():
            dias_mes = irradiacion_diaria_sin_ceros[
                irradiacion_diaria_sin_ceros.index.to_period('M') == fecha.to_period('M')]
            print(f"{fecha.strftime('%Y-%m')}: {row['Irradiación (kWh/m²/día)']:.3f} kWh/m²/día | "
                  f"{row['Irradiancia (W/m²)']:.2f} ± {row['Desv. Std. Irradiancia (W/m²)']:.2f} W/m² "
                  f"({len(dias_mes)} días)")
        
        mes_mas_soleado = estadisticas_mensuales['Irradiación (kWh/m²/día)'].idxmax()
        print(f"\nMes más soleado: {mes_mas_soleado.strftime('%B %Y')}")
        print(f"  • Irradiación: {estadisticas_mensuales.loc[mes_mas_soleado, 'Irradiación (kWh/m²/día)']:.3f} kWh/m²/día")
        print(f"  • Irradiancia: {estadisticas_mensuales.loc[mes_mas_soleado, 'Irradiancia (W/m²)']:.2f} ± "
              f"{estadisticas_mensuales.loc[mes_mas_soleado, 'Desv. Std. Irradiancia (W/m²)']:.2f} W/m²")
        
        # ==================== TOP 10 DÍAS ====================
        print("\n" + "="*70)
        print("TOP 10 DÍAS CON MAYOR IRRADIACIÓN")
        print("="*70)
        
        top_10 = irradiacion_diaria_sin_ceros.nlargest(10)
        for i, (fecha, valor) in enumerate(top_10.items(), 1):
            irradiancia_dia = irradiancia_promedio_diaria_sin_ceros.get(fecha, 0)
            print(f"{i:2d}. {fecha.date()}: {valor:.3f} kWh/m²/día ({irradiancia_dia:.2f} W/m²)")
        
        # Mostrar todas las gráficas
        plt.show()
        
        # ==================== RESUMEN FINAL ====================
        print("\n" + "="*70)
        print("RESUMEN FINAL DEL ANÁLISIS")
        print("="*70)
        print(f"Irradiación promedio diaria (GHI): {promedio_irradiacion:.3f} kWh/m²/día")
        print(f"Irradiancia promedio diaria: {promedio_irradiancia:.2f} ± {std_irradiancia:.2f} W/m²")
        print(f"Irradiancia máxima en curva diaria: {irradiancia_maxima:.2f} W/m² (hora {hora_maxima:.2f})")
        print("="*70)
        
        return {
            'promedio_irradiacion': promedio_irradiacion,
            'promedio_irradiancia': promedio_irradiancia,
            'std_irradiancia': std_irradiancia,
            'irradiancia_maxima_horaria': irradiancia_maxima,
            'hora_maxima': hora_maxima,
            'estadisticas_mensuales': estadisticas_mensuales,
            'top_10_dias': top_10
        }
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{nombre_archivo_csv}'.")
        return None
    except Exception as e:
        print(f"ERROR durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    archivo = r'C:\Users\JuanF\Downloads\datos_simplificados.csv'
    resultados = analisis_completo_irradiacion(archivo)