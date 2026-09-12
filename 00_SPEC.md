# 📐 SPEC & Blueprint: Deep Learning Model Deployment Sandbox (GP-029)
**Target Company:** Inetum | **Target Role:** Senior Data Scientist
**Perspective:** Causal & Survival Lifecycle Analytics (`CAUSAL_SURVIVAL`)
**Core Algorithm:** Kaplan-Meier & Cox Proportional Hazards Hazard Modeling

---

## 🏛️ 1. The Core Business Bottleneck
Inetum enfrenta una pérdida crítica de eficiencia operacional y visibilidad analítica debido a la falta de modelos predictivos y automatización de métricas sobre flujos de datos transaccionales masivos. Las soluciones tradicionales dependen de planillas manuales propensas a errores o arquitecturas rígidas de alto costo.

## ⚖️ 2. Architectural Trade-Offs Evaluated
- **Trade-Off Central:** Modelado Dinámico Temporal vs Agregaciones Estáticas Tradicionales
- **Alternativa Descartada 1:** Consultas directas no indexadas en bases transaccionales (Provocaba bloqueos y latencias >10s).
- **Alternativa Descartada 2:** Modelos de caja negra no interpretables (Rechazados por la dirección financiera por falta de explicabilidad).
- **Solución Adoptada:** Pipeline desacoplado con DuckDB en memoria + motor analítico especializado en Causal & Survival Lifecycle Analytics.

---

## 🎙️ 3. Guion de Preguntas Trampa para la Entrevista

### ❓ Pregunta Trampa 1: ¿Por qué decidiste aplicar Kaplan-Meier & Cox Proportional Hazards Hazard Modeling en lugar de una agregación tradicional en SQL?
> **💡 Respuesta Estratégica:** 
> *"Porque las agregaciones estáticas tradicionales solo miran el pasado y asumen comportamientos lineales. Al aplicar Kaplan-Meier & Cox Proportional Hazards Hazard Modeling, capturamos la dinámica probabilística y estocástica de los datos de Inetum, permitiendo a la dirección anticipar desviaciones y optimizar márgenes antes de que ocurra la pérdida."*

### ❓ Pregunta Trampa 2: ¿Cómo garantizas que el pipeline no se rompa si el volumen de datos se multiplica por 10?
> **💡 Respuesta Estratégica:**
> *"Diseñé la arquitectura con procesamiento columnar vectorizado (DuckDB/Parquet) y validación estricta de contratos de datos. Si el volumen crece 10x, el motor realiza streaming por micro-lotes sin sobrecargar la memoria RAM, manteniendo latencias de consulta sub-segundo."*

### ❓ Pregunta Trampa 3: ¿Cómo comunicas estos resultados técnicos a stakeholders no técnicos?
> **💡 Respuesta Estratégica:**
> *"Traduzco la matemática en tres indicadores clave de negocio (Impacto Financiero, Tiempo Ahorrado y Nivel de Confianza). Además, automatizo tableros ejecutivos en Power BI/Excel donde los directores pueden ver los escenarios en 2 segundos sin lidiar con la complejidad interna."*
