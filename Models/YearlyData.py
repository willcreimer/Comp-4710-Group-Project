import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data-cleaned.csv")
data['fire_year'] = pd.to_numeric(data['fire_year'], errors='coerce')
yearly_analysis = data.groupby(['fire_year', 'isNaturalCaused']).size().unstack(fill_value=0)
yearly_analysis.columns = ['Not Naturally Caused', 'Naturally Caused']
print(yearly_analysis)


yearly_analysis.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Wildfire Causes by Year")
plt.xlabel("Year")
plt.ylabel("Number of Fires")
plt.xticks(rotation=45)
plt.legend(title="Cause", labels=["Not Naturally Caused", "Naturally Caused"])
plt.show()
