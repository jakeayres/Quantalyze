import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import quantalyze as qa



@st.dialog("Upload Files")
def upload_files():

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    if uploaded_files:
        if st.button('Load files'):
            st.session_state['dfs'] = [pd.read_csv(file) for file in uploaded_files]
            st.session_state['files'] = [file.name for file in uploaded_files]
            for file in uploaded_files:
                st.toast(f'{file.name} loaded')
            st.rerun()
    else:
        st.info('Upload files')


def get_x_column():
    x_column = st.selectbox('x Column', st.session_state['dfs'][0].columns, key='active_x_column')
    return x_column

def get_y_column():
    y_column = st.selectbox('y Column', st.session_state['dfs'][0].columns, key='active_y_column')
    return y_column


def main():
    
    st.title('Resistivity Analysis')

    if st.button('Replace Files', type='primary'):
        upload_files()


    if st.session_state.get('dfs'):

        for f in st.session_state['files']:
            st.success(f)

        st.header('Raw Data')
        cols = st.columns(2, gap='large')
        with cols[0]:
            x_column = get_x_column()
            y_column = get_y_column()
        with cols[1]:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            for df in st.session_state['dfs']:
                ax.plot(df[x_column], df[y_column])

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            fig.tight_layout()
            st.pyplot(fig)



        st.header('Combine Data')
        cols = st.columns(2, gap='large')
        df = pd.concat(st.session_state['dfs'])
        df.rename(columns={x_column: 'temperature', y_column: 'voltage'}, inplace=True)
        df = df[['temperature', 'voltage']]
        with cols[0]:
            minimum = st.number_input('Minimum', value=df['temperature'].min(), step=1e-3, format='%.3f')
            maximum = st.number_input('Maximum', value=df['temperature'].max(), step=1e-3, format='%.3f')
            step = st.number_input('Step', value=0.1, step=1e-3, format='%.3f')

        with cols[1]:
            df = qa.bin(df, 'temperature', minimum, maximum, step)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(df['temperature'], df['voltage'])
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Voltage (V)')
            fig.tight_layout()
            st.pyplot(fig)


        st.header('Convert to Resistivity')
        cols = st.columns(2, gap='large')

        with cols[0]:
            current = st.number_input('Current (mA)', value=1.0, step=1e-3, format='%.3f')
            length = st.number_input('Length (um)', value=1.0, step=1e-3, format='%.3f')
            width = st.number_input('Width (um)', value=1.0, step=1e-3, format='%.3f')
            thickness = st.number_input('Thickness (um)', value=1.0, step=1e-3, format='%.3f')

        with cols[1]:
            df['resistance'] = qa.calculate_resistance(df, voltage='voltage', current=current*1e-3)
            df['resistivity'] = qa.calculate_resistivity(df, resistance='resistance', length=length*1e-6, width=width*1e-6, thickness=thickness*1e-6)
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(df['temperature'], df['resistivity']*1e8)
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel(r'Resistivity ($\mu \Omega$ cm)')
            fig.tight_layout()
            st.pyplot(fig)


        st.header('Save Data')
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='processed_data.data',
            mime='text/csv',
            on_click=lambda: st.success('Download started!'),
            type='primary',
        )
            

    else:
        st.info('Upload files')


if __name__ == "__main__":
    main()