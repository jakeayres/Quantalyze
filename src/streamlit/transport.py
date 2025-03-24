import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import quantalyze as qa


st.set_page_config(page_title='Transport GUI', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)


def fetch_files():
    st.header('Select data files')
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    if uploaded_files:
        if st.button('Load files'):
            st.session_state['dfs'] = [pd.read_csv(file) for file in uploaded_files]
            for file in uploaded_files:
                st.toast(f'{file.name} loaded')


    if st.session_state.get('dfs'):
        return st.session_state['dfs']
    else:
        st.info('Upload files')
        raise Exception('No files uploaded')



def main():

    cols = st.columns(2, gap='large')
    with cols[0]:
        st.title('Visualization')
    with cols[1]:
        st.title('Processing')

    try:
        with cols[0]:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            plot = st.pyplot(fig)
            data = st.dataframe(pd.DataFrame())
    except Exception as e:
        st.error(e)
        return 0
    
    try:
        with cols[1]:
            raw_dfs = fetch_files()
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0
    

    try:
        with cols[1]:
            st.header('Select columns')
            x_column = st.selectbox('x Column', raw_dfs[0].columns, key='active_x_column')
            
            col1, col2 = st.columns(2)
            with col1:
                symm_toggle = st.checkbox('Symmetrize Column', key='symm_toggle')
            with col2:
                symmetrize_column = st.selectbox('Symmetrize Column', raw_dfs[0].columns, key='active_symmetrize_column') if symm_toggle else None
            
            col3, col4 = st.columns(2)
            with col3:
                antisymm_toggle = st.checkbox('Antisymmetrize Column', key='antisymm_toggle')
            with col4:
                antisymmetrize_column = st.selectbox('Antisymmetrize Column', raw_dfs[0].columns, key='active_antisymmetrize_column') if antisymm_toggle else None
            
            
            if symmetrize_column:
                df1 = qa.symmetrize(raw_dfs, x_column, symmetrize_column, -14, 14, 0.1)
                df1.rename(columns={x_column: 'field'}, inplace=True)
                df1.rename(columns={symmetrize_column: 'voltage'}, inplace=True)
            if antisymmetrize_column:
                df2 = qa.antisymmetrize(raw_dfs, x_column, antisymmetrize_column, -14, 14, 0.1)
                df2.rename(columns={x_column: 'field'}, inplace=True)
                df2.rename(columns={antisymmetrize_column: 'hall_voltage'}, inplace=True)
            
            if symmetrize_column and antisymmetrize_column:
                df = pd.merge(df1, df2, on='field', how='outer')
            elif symmetrize_column:
                df = df1
            elif antisymmetrize_column:
                df = df2
            else:
                df = pd.DataFrame()

            with data:
                st.dataframe(df)
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0

    try:
        with plot:
            if symmetrize_column:
                for raw_df in raw_dfs:
                    ax[0, 0].plot(raw_df[x_column], raw_df[symmetrize_column], color='tab:red', alpha=0.5)
                    ax[0, 0].plot(raw_df[x_column]*-1, raw_df[symmetrize_column], color='tab:blue', alpha=0.5)
                    st.pyplot(fig)
                ax[0, 0].plot(df['field'], df['voltage'], '-', color='black')
                st.pyplot(fig)

            if antisymmetrize_column:
                for raw_df in raw_dfs:
                    ax[1, 0].plot(raw_df[x_column], raw_df[antisymmetrize_column], color='tab:red', alpha=0.5)
                    ax[1, 0].plot(raw_df[x_column]*-1, raw_df[antisymmetrize_column], color='tab:blue', alpha=0.5)
                    st.pyplot(fig)
                ax[1, 0].plot(df['field'], df['hall_voltage'], '-', color='black')
                st.pyplot(fig)
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0

    
    try:
        with cols[1]:
            st.header('Current')
            current = st.number_input('Current (mA)', value=None, step=1e-3, format="%0.3f")
        if current:

            if symmetrize_column:
                df['resistance'] = qa.calculate_resistance(df, voltage='voltage', current=current*1e-3)
                with plot:
                    ax[0, 1].clear()
                    ax[0, 1].plot(df['field'], df['resistance'], '-', color='black')
                    st.pyplot(fig)

            if antisymmetrize_column:
                df['hall_resistance'] = qa.calculate_resistance(df, voltage='hall_voltage', current=current*1e-3)
                with plot:
                    ax[1, 1].clear()
                    ax[1, 1].plot(df['field'], df['hall_resistance'], '-', color='black')
                    st.pyplot(fig)
            
            with data:
                st.dataframe(df)
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0


    try:
        with cols[1]:
            st.header('Sample geometry')
            col1, col2 = st.columns(2)
            with col1:
                geometric_factor = st.number_input('Geometric factor', value=None, step=1e-6, format="%0.9f")
            with col2:
                length = st.number_input('Length (um)', value=None, step=1e-3, format="%0.3f")
                width = st.number_input('Width (um)', value=None, step=1e-3, format="%0.3f")
                thickness = st.number_input('Thickness (um)', value=None, step=1e-3, format="%0.3f")
            
            if length and width and thickness and symmetrize_column:
                st.info('Using length, width, and thickness')
                df['resistivity'] = qa.calculate_resistivity(df, resistance='resistance', length=length*1e-6, width=width*1e-6, thickness=thickness*1e-6)
            elif geometric_factor:
                st.info('Using geometric factor')
                df['resistivity'] = qa.calculate_resistivity(df, resistance='resistance', geometric_factor=geometric_factor)
            else:
                st.warning('Full geometry needed to compute resistivity')

            if thickness and antisymmetrize_column:
                df['hall_resistivity'] = qa.calculate_hall_resistivity(df, hall_resistance='hall_resistance', thickness=thickness*1e-6)
            else:
                st.warning('Thickness needed to compute Hall resistivity')
            
        with data:
            st.dataframe(df)
        
        if symmetrize_column:
            with plot:
                ax[0, 1].clear()
                ax[0, 1].plot(df['field'], df['resistivity']*1e8, '-', color='black')
                st.pyplot(fig)

    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0
    

    try:
        with cols[1]:
            st.header('Extra columns')
            if 'new_columns' not in st.session_state:
                st.session_state.new_columns = []

            if st.button('Add new column'):
                st.session_state.new_columns.append({'name': '', 'value': 0.0})

            for i, col in enumerate(st.session_state.new_columns):
                col1, col2 = st.columns(2)
                with col1:
                    col['name'] = st.text_input(f'New column name {i+1}', value=col['name'])
                with col2:
                    col['value'] = st.number_input(f'New column value {i+1}', value=col['value'], step=1.0, format="%0.2f")

        for col in st.session_state.new_columns:
            if col['name']:
                df.insert(1, col['name'], col['value'])

        with data:
            st.dataframe(df)
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0
    


    try:
        with cols[1]:
            st.header('Save data')
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='processed_data.data',
                mime='text/csv',
                on_click=lambda: st.success('Download started!')
            )
    except Exception as e:
        st.error(e)
        st.exception(e)
        return 0



if __name__ == "__main__":
    main()