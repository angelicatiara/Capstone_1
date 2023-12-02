import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# st.set_page_config(page_title= "Retail Platform",page_icon='logo.png')
def get_customer_rfm(customer_id, df):
    # Calculate RFM values and cluster memberships for the customer
    # This is placeholder logic; you will need to replace it with your actual calculation logic
    customer_data = df[df['CustomerID'] == customer_id]
    customer_data['InvoiceDate'] = pd.to_datetime(customer_data['InvoiceDate'])

    rfm_values = {
        'total_purchases': customer_data['InvoiceNo'].nunique(),
        'total_spent': customer_data['TotalCost'].sum(),
        'recency':customer_data['Recency'].min(),
        # clusters
    }
    return rfm_values
# Function to load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('OnlineRetail.csv')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])  # Convert InvoiceDate to datetime
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']  # Calculate total price
    return data

data = load_data()
df_uk =pd.read_csv('df_uk.csv')
df_rfm=pd.read_csv('df_rfm.csv')
st.markdown("""
        <style>
            .main_container {
                background-color: #FFFFFF;
            }

            h1 {
                text-align: center;
                color: #FF3131;
            }
            h2{
                text-align: center;
            }
            .sidebar .sidebar-content {
                background-color: #F2F2F2;
            }
            
            .sidebar .sidebar-button{
                 color: #ffffff;
                background-color: #FF3131;
                border: none;
                border-radius: 4px;
                padding: 0.75rem 1.5rem;
                margin: 0.75rem 0;
                position: absolute;
                left: 10% ;
            }
            
            .stButton>button {
                color: #ffffff;
                background-color: #FF3131;
                border: none;
                border-radius: 4px;
                padding: 0.75rem 1.5rem;
                margin: 0.75rem 0;
                position: absolute;
                left:40%;

            }
            .stButton>button:hover {
                background-color:  #FF4B4B;
                text-align: center;
                color: #FFFFFF;
            }
           
            
          
            body {
            background-color: #F2F2F22;}
        </style>
    """, unsafe_allow_html=True)

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False
    

# Define the welcome page form
if not st.session_state['submitted']:
    
        # st.write("Welcome to Eng.Majed AutoMobile Shop! Please submit to continue.")
        col1,col2,col3 = st.columns([0.2,0.4,0.2])
        col2.image("logo.png",width=300,)
        st.markdown("----", unsafe_allow_html=True)
        st.subheader("Welcome to..")
        st.title('Retaill store Platform')
        st.subheader("The PERFECT place to find goods prices with the touch of machine learning")
        st.markdown("----", unsafe_allow_html=True)
        submitted = st.button("Let's GO")
        if submitted:
            st.session_state['submitted'] = True
            st.experimental_rerun()  # Rerun the app to update the state

# Define your tabs
if st.session_state['submitted']:
    tab1, tab2,tab3 = st.tabs(["Clusters", "RFM Details","Visualization :sunglasses:"])

    with tab3:
                
        # Streamlit app layout
        st.title('Customer Data Visualization')

        # Dropdown to select a customer ID
        customer_ids = data['CustomerID'].dropna().unique()
        customer_ids=sorted(customer_ids)
        selected_customer_id = st.selectbox('Select a Customer ID', customer_ids)

        # Filter data based on selected customer ID
        customer_data = data[data['CustomerID'] == selected_customer_id]

        # Purchase History Timeline
        st.subheader('Purchase History Timeline')
        if not customer_data.empty:
            timeline_data = customer_data.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
            plt.figure(figsize=(10, 4))
            plt.style.use('dark_background')
            sns.set_style("darkgrid")
            sns.lineplot(x='InvoiceDate', y='TotalPrice', data=timeline_data, color='red')  # Set line color to white for visibility
            plt.xticks(rotation=45 ,color='white')
            plt.yticks(rotation=0,color='white')
            plt.xlabel('Invoice Date',color='white')
            plt.ylabel('Total Spent',color='white')
            plt.title('Total Spending Over Time',color='white')
            st.pyplot(plt)
        else:
            st.write("No data available for this customer.",color='white')

        # Spending Analysis
        st.subheader('Spending Analysis')
        if not customer_data.empty:
            product_data = customer_data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=product_data.values, y=product_data.index, palette='viridis')
            plt.xlabel('Total Spent',color='white')
            plt.xticks(rotation=0 ,color='white')
            plt.yticks(rotation=0,color='white')
            plt.ylabel('Product',color='white')
            plt.title('Top 10 Products by Spending',color='white')
            st.pyplot(plt)
        else:
            st.write("No data available for this customer.")

        # Run this app with 'streamlit run app.py'
    # Inside the 'Details' tab in your Streamlit app
    with tab2:
        st.title("RFM Analysis Details")
        frequency_data = df_uk.groupby('CustomerID')['Frequency'].max()

        # Create a histogram using Plotly
        fig = px.histogram(frequency_data, nbins=200)
        fig.update_layout(
            title_text='Frequency Value Distribution',
            xaxis_title_text='Frequency',
            yaxis_title_text='Count',
            bargap=0.2,  # Gap between bars
            xaxis = dict(range=[-10, 60])
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        monetary_data = df_uk.groupby('CustomerID')['Monetary'].max()

        # Create a histogram using Plotly
        fig = px.histogram(monetary_data, nbins=400)
        fig.update_layout(
            title_text='Monetary Value Distribution',
            xaxis_title_text='Monetary',
            yaxis_title_text='Count',
            bargap=0.2,  # Gap between bars
            xaxis = dict(range=[-10000, 40000])
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        recency_data = df_uk.groupby('CustomerID')['Recency'].max()

        # Create a histogram using Plotly
        fig = px.histogram(recency_data, nbins=80)
        fig.update_layout(
            title_text='Recency Value Distribution',
            xaxis_title_text='Recency',
            yaxis_title_text='Count',
            bargap=0.2  # Gap between bars
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        total_cost_per_customer = df_uk.groupby('CustomerID')['TotalCost'].sum()
        # # Filter out customers with a TotalCost of 0
        good_customers = total_cost_per_customer[total_cost_per_customer != 0].keys()
        # selected_customer_id1 = st.selectbox('Select a Customer', good_customers)
        # cost = df_uk[df_uk['CustomerID'] ==selected_customer_id1]['TotalCost'].sum()
        # st.title(cost)
        st.title('Customer Insights')
        st.write('customer behaviour')


        # Dropdown to select customer ID
        customer_ids = df_uk['CustomerID'].unique()
        selected_customer_id2 = st.selectbox('Select a customer to show more information about him.', good_customers)

        # Display user expenses plot
        expenses_plot_data = df_uk[df_uk['CustomerID'] == selected_customer_id2].groupby('InvoiceDate')['TotalCost'].sum().reset_index()
        fig = px.line(expenses_plot_data, x='InvoiceDate', y='TotalCost', title='User Expenses')
        st.plotly_chart(fig)

        # Get and display RFM statistics
        customer_rfm = get_customer_rfm(selected_customer_id2, df_uk)
        st.write(f"The user purchased {customer_rfm['total_purchases']} times.")
        st.write(f"Customer spent {customer_rfm['total_spent']} dollars in total.")
        st.write(f"The last time Customer bought something was {customer_rfm['recency']} days ago.")
        # st.write("She/he belongs to the clusters:")
        # st.write(f"- Revenue_cluster = {customer_rfm['revenue_cluster']}")
        # st.write(f"- Frequency_cluster = {customer_rfm['frequency_cluster']}")
        # st.write(f"- Recency_cluster = {customer_rfm['recency_cluster']}")

        # Determine and display customer description
        # This is placeholder text; you will need to implement the logic to determine the customer description
        customer_description = f"Customer level description is : {df_rfm[df_rfm['Unnamed: 0']==selected_customer_id2]['RFM_Scores_Levels'].values[0]}"
        st.write(f"The customer can be described as: {customer_description}")
        # st.write(df_rfm)