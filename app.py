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
            fig = px.line(
    timeline_data,
    x='InvoiceDate',
    y='TotalPrice',
    title='Total Spending Over Time',
    markers=True,  # Include markers for each point
            )

            # Customize the chart for a dark background with white text
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                font_color='white',
                xaxis=dict(
                    title='Invoice Date',
                    showgrid=True,
                    gridcolor='grey',
                    tickangle=45  # Rotate the x-axis labels
                ),
                yaxis=dict(
                    title='Total Spent',
                    showgrid=True,
                    gridcolor='grey'
                ),
            )

            # Show the figure in an interactive environment, such as Jupyter Notebook
            # fig.show()

            # To display the chart in Streamlit, use the following:
            st.plotly_chart(fig)
        else:
            st.write("No data available for this customer.",color='white')

        # Spending Analysis
        st.subheader('Spending Analysis')
        if not customer_data.empty:
            product_data = customer_data.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(
            product_data, 
                y='Description', 
                x='TotalPrice', 
                title='Top 10 Products by Spending', 
                orientation='h',  # This makes it a horizontal bar chart
                color='TotalPrice',  # This will use the 'TotalPrice' to color the bars
                color_continuous_scale=px.colors.sequential.Viridis  # This sets the color scale to Viridis
            )

            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                font=dict(color='white'),  # White font color
            )

            # Show the figure
            # fig.show()

            # If you're using Streamlit, replace fig.show() with:
            st.plotly_chart(fig)
        else:
            st.write("No data available for this customer.")

        
        hola = df_uk[df_uk['CustomerID'] == selected_customer_id].groupby('PriceRange').agg({'TotalCost': 'sum'}).reset_index()
        
# Create a bar plot using Plotly
        fig = px.bar(hola, x='PriceRange', y='TotalCost', title='Sales by Product Price Range',
                    labels={'TotalCost': 'Total Sales', 'PriceRange': 'Price Range (£)'})

        # In a Streamlit app, you would use this to display the plot
        st.plotly_chart(fig)
        st.write(hola,selected_customer_id)

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