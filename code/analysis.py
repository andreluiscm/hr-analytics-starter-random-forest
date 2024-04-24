import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


employees_df = pd.read_csv(filepath_or_buffer='./data/people_analytics_start.csv')

employees_df['education'].replace([np.nan, "Associate's degree", "Bachelor's degree", "Master's degree"], [0, 1, 2, 3], inplace=True)

bins = [20,25,30,35,40,45,50,55,60,65]
labels = ['[20-24]','[25-29]','[30-34]','[35-39]','[40-44]','[45-49]','[50-54]','[55-59]','[60-64]']
employees_df['age_group'] = pd.cut(employees_df['age'], bins=bins, labels=labels, right=False)

left_df = employees_df[employees_df['active_status'] == 0]
stayed_df = employees_df[employees_df['active_status'] == 1]



# Tipo do desligamento
plot = sns.histplot(left_df['term_type'], shrink=0.8)

# Annotate each bar with its value
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height() - 5), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

plot.set_xlabel('Tipo de desligamento do colaborador (term_type)')
plot.set_ylabel('# Ocorrências')
plot.set_xticklabels(['Voluntário', 'Involuntário'])
plt.tight_layout()
plt.savefig('./figures/analysis_term_type.png')
plt.close()



# Motivo do desligamento
plot = sns.histplot(left_df['term_reason'], shrink=0.8)

# Annotate each bar with its value
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height() - 1), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

plot.set_xlabel('Motivo do desligamento do colaborador (term_reason)')
plot.set_ylabel('# Ocorrências')
plot.set_xticklabels(['Oportunidade melhor', 'Mudança de carreira', 'Desempenho', 'Salário melhor',
                      'Benefícios mais flexíveis', 'Razões pessoais', 'Rescisão por justa causa', 'Realocação',
                      'Reestruturação da empresa', 'Cortes de orçamento'])
plot.figure.set_size_inches(20, 8)
plt.tight_layout()
plt.savefig('./figures/analysis_term_reason.png')
plt.close()



# Modelo de trabalho
plt.figure(figsize=(12, 6))
x, y, hue = "location", "proportion", "active_status"

(employees_df[x]
 .groupby(employees_df[hue])
 .value_counts(normalize=True)
#  .rename(y)
 .reset_index()
 .pipe(sns.barplot, x=x, y=y, hue=hue, dodge=0.25)
)

plt.xlabel('Modelo de trabalho (location)')
plt.ylabel('Proporção')
plt.xticks(['Remote', 'On-site'], ['Remoto', 'Presencial'])

# Add legend
legend = plt.legend(title='Situação do colaborador')

# Edit legend text
legend.texts[0].set_text('Inativo')
legend.texts[1].set_text('Ativo')

ax = plt.gca()
# Annotate each bar with its value
for p in ax.patches:
    if p._height > 0:
        ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('./figures/analysis_location.png')
plt.close()



# Tenure
plt.figure(figsize=(12, 12))
sns.scatterplot(data=employees_df, x="tenure", y="age", hue="active_status")

plt.xlabel('Tempo de serviço do colaborador (tenure)')
plt.ylabel('Idade do colaborador')

# Add legend
legend = plt.legend(title='Situação do colaborador')

# Edit legend text
legend.texts[0].set_text('Inativo')
legend.texts[1].set_text('Ativo')

plt.tight_layout()
plt.savefig('./figures/analysis_tenure.png')
plt.close()



# Gênero
plt.figure(figsize=(25, 8))
plt.yscale("log")
x, y, hue = "gender", "proportion", "active_status"

(employees_df[x]
 .groupby(employees_df[hue])
 .value_counts(normalize=True)
#  .rename(y)
 .reset_index()
 .pipe(sns.barplot, x=x, y=y, hue=hue, dodge=0.25)
)

plt.xlabel('Gênero do colaborador (gender)')
plt.ylabel('Proporção')
plt.xticks(['Male', 'Female', 'Other'], ['Masculino', 'Feminino', 'Outro'])

# Add legend
legend = plt.legend(title='Situação do colaborador')

# Edit legend text
legend.texts[0].set_text('Inativo')
legend.texts[1].set_text('Ativo')

ax = plt.gca()
# Annotate each bar with its value
for p in ax.patches:
    if p._height > 0:
        ax.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('./figures/analysis_gender.png')
plt.close()



# Grau de educação
plt.figure(figsize=(25, 8))
plt.yscale("log")
x, y, hue = "education", "proportion", "active_status"

(employees_df[x]
 .groupby(employees_df[hue])
 .value_counts(normalize=True)
#  .rename(y)
 .reset_index()
 .pipe(sns.barplot, x=x, y=y, hue=hue, dodge=0.25)
)

plt.xlabel('Grau de educação do colaborador (education)')
plt.ylabel('Proporção')
plt.xticks([0, 1, 2, 3], ['Outro', 'Nível Técnico', 'Nível Superior', 'Mestrado'])

# Add legend
legend = plt.legend(title='Situação do colaborador')

# Edit legend text
legend.texts[0].set_text('Inativo')
legend.texts[1].set_text('Ativo')

ax = plt.gca()
# Annotate each bar with its value
for p in ax.patches:
    if p._height > 0:
        ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig('./figures/analysis_education.png')
plt.close()
