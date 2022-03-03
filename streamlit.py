import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from interactive_cube import Cube

#the body of the page
def main():
    st.title('Reinforced Learning with Rubik\'s Cubes')
    #making cube
    c = Cube(3)

    option = st.selectbox('Which face would you like to move?',
    ('L', 'L\'', 'R', 'R\'', 'U', 'U\'', 'D', 'D\'', 'F' , 'F\'', 'B', 'B\''))

    if len(option) > 1:
        c.rotate_face(option[0], -1)
    else:
        c.rotate_face(option)

    if st.button('Solve Cube'):
        c = Cube(3)

    fig = c.draw_interactive()
    st.pyplot(fig)

# the controller
def load_page():
    main()

if __name__ == "__main__":
    load_page()
