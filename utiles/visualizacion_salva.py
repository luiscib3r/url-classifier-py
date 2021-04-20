# 2. Data Visualization esto no va en la app esto es para sacar las graficas del dataset generado

# # Heatmap todo ver como exportar imagen
# corrmat = urldata.corr()
# f, ax = plt.subplots(figsize=(25, 19))
# sns.heatmap(corrmat, square=True, annot=True, annot_kws={'size': 10})
#
# # Bar
# plt.figure(figsize=(15, 5))
# sns.countplot(x='label', data=urldata)
# plt.title("Count Of URLs", fontsize=20)
# plt.xlabel("Type Of URLs", fontsize=18)
# plt.ylabel("Number Of URLs", fontsize=18)
#
# print("Percent Of Malicious URLs:{:.2f} %".format(
#     len(urldata[urldata['label'] == 'malicious']) / len(urldata['label']) * 100))
# print("Percent Of Benign URLs:{:.2f} %".format(
#     len(urldata[urldata['label'] == 'benign']) / len(urldata['label']) * 100))

# plt.figure(figsize=(20, 5))
# plt.hist(urldata['url_length'], bins=50, color='LightBlue')
# plt.title("URL-Length", fontsize=20)
# plt.xlabel("Url-Length", fontsize=18)
# plt.ylabel("Number Of Urls", fontsize=18)
# plt.ylim(0, 1000)

# plt.figure(figsize=(20, 5))
# plt.hist(urldata['hostname_length'], bins=50, color='Lightgreen')
# plt.title("Hostname-Length", fontsize=20)
# plt.xlabel("Length Of Hostname", fontsize=18)
# plt.ylabel("Number Of Urls", fontsize=18)
# plt.ylim(0, 1000)
#
# plt.figure(figsize=(20, 5))
# plt.hist(urldata['tld_length'], bins=50, color='Lightgreen')
# plt.title("TLD-Length", fontsize=20)
# plt.xlabel("Length Of TLD", fontsize=18)
# plt.ylabel("Number Of Urls", fontsize=18)
# plt.ylim(0, 1000)
#
# plt.figure(figsize=(15, 5))
# plt.title("Number Of Directories In Url", fontsize=20)
# sns.countplot(x='count_dir', data=urldata)
# plt.xlabel("Number Of Directories", fontsize=18)
# plt.ylabel("Number Of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Number Of Directories In Url", fontsize=20)
# sns.countplot(x='count_dir', data=urldata, hue='label')
# plt.xlabel("Number Of Directories", fontsize=18)
# plt.ylabel("Number Of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of IP In Url", fontsize=20)
# plt.xlabel("Use Of IP", fontsize=18)
#
# sns.countplot(urldata['use_of_ip'])
# plt.ylabel("Number of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of IP In Url", fontsize=20)
# plt.xlabel("Use Of IP", fontsize=18)
# plt.ylabel("Number of URLs", fontsize=18)
# sns.countplot(urldata['use_of_ip'], hue='label', data=urldata)
# plt.ylabel("Number of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of http In Url", fontsize=20)
# plt.xlabel("Use Of IP", fontsize=18)
# plt.ylim((0, 1000))
# sns.countplot(urldata['count-http'])
# plt.ylabel("Number of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of http In Url", fontsize=20)
# plt.xlabel("Count Of http", fontsize=18)
# plt.ylabel("Number of URLs", fontsize=18)
# plt.ylim((0, 1000))
# sns.countplot(urldata['count-http'], hue='label', data=urldata)
# plt.ylabel("Number of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of http In Url", fontsize=20)
# plt.xlabel("Count Of http", fontsize=18)
#
# sns.countplot(urldata['count-http'], hue='label', data=urldata)
#
# plt.ylabel("Number of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of WWW In URL", fontsize=20)
# plt.xlabel("Count Of WWW", fontsize=18)
# sns.countplot(urldata['count-www'])
# plt.ylim(0, 1000)
# plt.ylabel("Number Of URLs", fontsize=18)
#
# plt.figure(figsize=(15, 5))
# plt.title("Use Of WWW In URL", fontsize=20)
# plt.xlabel("Count Of WWW", fontsize=18)
#
# sns.countplot(urldata['count-www'], hue='label', data=urldata)
# plt.ylim(0, 1000)
# plt.ylabel("Number Of URLs", fontsize=18)