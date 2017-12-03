# DataAnalysis_HTML.py
# 
# Copyright (C) 2017  Yangang Chen
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# 
# 
# 
# Process and analyze the html data, downloaded from
# http://www.investing.com/currencies/gbp-usd-historical-data
# http://www.investing.com/commodities/gold-historical-data

################################

import urllib, bs4  # bs4 (Beautiful Soup) is a Python library for pulling data out of HTML and XML files
import pandas as pd
import os
import matplotlib.pyplot as plt


def html2txt(html_name):
    ## Converting html to txt file
    html_fullname = ("file://" + os.getcwd() + "/DataAnalysis_HTML/" + html_name + ".html").replace(' ', '%20')
    print(html_fullname)
    html = urllib.request.urlopen(html_fullname).read()

    ## Untimate reformatting of the html file
    processed_html = bs4.BeautifulSoup(html, "html.parser")
    # print(processed_html)

    ## Save the processed html file into a txt file
    txt_fullname = "DataAnalysis_HTML/" + html_name + ".txt"
    with open(txt_fullname, "w") as file:
        ## Find the section with the id "curr_table" (which corresponds to the table we want to extract)
        for content in processed_html.find_all(id="curr_table"):
            # print(content.text)
            file.write(content.text)


def txt2csv(file_name):
    ## Read the txt file
    doc = []
    txt_fullname = "DataAnalysis_HTML/" + file_name + ".txt"
    with open(txt_fullname, "r") as file:
        for line in file:
            doc.append(line)
    # print(doc)

    ## Construct a pandas Series
    df = pd.Series(doc)
    # print(df)

    ## Remove all the '\n' at the end of each line and all the ',' in the numerical values
    df = df.str.rstrip().str.replace(',', '')
    ## Record the column names
    column_size = 8
    column_names = df[3:1 + column_size].values
    ## Strip the first few lines of the txt file
    df = df[len(df) - (len(df) // column_size - 1) * column_size:]
    ## Reshape the data from the txt file
    df = df.values.reshape((len(df) // column_size, column_size))[:, :-2]

    ## Construct a pandas DataFrame
    df = pd.DataFrame(data=df, columns=column_names)
    ## Convert the column Date to the standard format
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # print(df)

    ## Save the file to csv
    csv_fullname = "DataAnalysis_HTML/" + file_name + ".csv"
    df.to_csv(csv_fullname)


def csv2plots():
    ## Read the csv files
    df_money = pd.read_csv("DataAnalysis_HTML/GBP USD Historical Data - Investing.com.csv",
                           index_col=0)
    df_money.index = pd.to_datetime(df_money.index)
    df_metal = pd.read_csv("DataAnalysis_HTML/Gold Futures Historical Prices - Investing.com.csv",
                           index_col=0)
    df_metal.index = pd.to_datetime(df_metal.index)
    # print(df_money.head())
    # print(df_metal.head())

    ## Plot the data
    fig = plt.figure(figsize=(10, 6))
    ts_money = df_money['Price'] * 1000
    ts_metal = df_metal['Price']
    ts_money.plot(label='Price (USD) of 1000 GBP', color="b")
    ts_metal.plot(label='Price (USD) of 1 Gold', color="r")
    plt.gcf().autofmt_xdate()  # beautify the x-labels
    plt.title("Historical Prices from Investing.com")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show()
    fig.savefig("DataAnalysis_HTML/Figure.pdf")


################################

html_name = "GBP USD Historical Data - Investing.com"
html2txt(html_name)
file_name = html_name
txt2csv(file_name)

html_name = "Gold Futures Historical Prices - Investing.com"
html2txt(html_name)
file_name = html_name
txt2csv(file_name)

csv2plots()
