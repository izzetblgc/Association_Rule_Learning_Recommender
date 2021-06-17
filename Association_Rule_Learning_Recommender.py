import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

### Veri Ön İşleme Adımları ###

#Eksik gözlemleri çıkaralım
df.dropna(inplace=True)

#İade edilen ürünleri çıkaralım
df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

### Aykırı değerleri belirlediğimiz limitlere getirelim ###

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T


### Veri setinden Germany müşterilerini seçelim. ###

# Postage verilen kargo ücretini belirtmektedir. Birliktelik kurallarını oluştururken diğer verileri baskılayabileceği
# için çıkarıyoruz.

df_ger = df[(df['Country'] == "Germany") & (df['Description'] != "POSTAGE")]

#Bu fonksiyon ile ürünlerin satın alınmış olmasını 1, satın alınmamış olmasını 0 ile temsil ederek bir pivot tablosu oluşturuyoruz

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

df_ger_inv_pro = create_invoice_product_df(df_ger, id=True)

### Apriori algoritmasıyla birliktelik kurallarının oluşturulması ###

frequent_itemsets = apriori(df_ger_inv_pro, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(20)

rules_ger = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

#StockCode ile ürün ismine erişebildiğimiz fonksiyon
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

### Sepete ekleyeceğimiz ürünler ###
check_id(df_ger, 21987)
#PACK OF 6 SKULL PAPER CUPS

check_id(df_ger, 23235)
#STORAGE TIN VINTAGE LEAF

check_id(df_ger, 22747)
#POPPY'S PLAYHOUSE BATHROOM

### Sepetteki ürünler için ürün önerisi ###

sorted_rules = rules_ger.sort_values("lift", ascending=False)


#Ürün önerisi yapacak olan fonksiyon

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


    return set(recommendation_list[0:rec_count]) #önerilerin tekrarlamaması için çıktıyı sete çevirdim

cart_1_recommendation = arl_recommender(sorted_rules, 21987, 5) #1. sepet önerileri
cart_2_recommendation = arl_recommender(sorted_rules, 23235, 5) #2. sepet önerileri
cart_3_recommendation = arl_recommender(sorted_rules, 22747, 5) #3. sepet önerileri

### Önerilen Ürünlerin İsimleri ###

def recommendation_products (cart_recommend):
    for i in cart_recommend:
       recommend_list =  check_id(df_ger,i)
    return recommend_list

# Sepetteki Ürün: (PACK OF 6 SKULL PAPER CUPS)

recommendation_products(cart_1_recommendation)

### 1.sepet önerilen ürünler ###
#PACK OF 6 SKULL PAPER PLATES
#PACK OF 20 SKULL PAPER NAPKINS
#SET/6 RED SPOTTY PAPER CUPS


# Sepetteki Ürün: STORAGE TIN VINTAGE LEAF

recommendation_products(cart_2_recommendation)

### 2.sepet önerilen ürünler ###

#ROUND STORAGE TIN VINTAGE LEAF
#SET OF TEA COFFEE SUGAR TINS PANTRY
#SET OF 4 KNICK KNACK TINS DOILEY


# Sepetteki Ürün: POPPY'S PLAYHOUSE BATHROOM

recommendation_products(cart_3_recommendation)

### 3.sepet önerilen ürünler ###
#["POPPY'S PLAYHOUSE BEDROOM "]
#["POPPY'S PLAYHOUSE LIVINGROOM "]














