import streamlit as st
from green_mcp_client import MCPClient

st.title("Green MCP Client (via MCPClient)")

server_url = st.text_input("MCP Server URL", "http://localhost:8000")
client = MCPClient(server_url)

st.header("Poem Generator")
theme = st.text_input("Theme for Poem", "nature")
if st.button("Generate Poem"):
    try:
        result = client.generate_poem(theme)
        st.success(result)
    except Exception as e:
        st.error(str(e))

st.header("Weather Batch")
cities = st.text_area("Cities (comma separated)", "London,Paris,Berlin")
if st.button("Get Weather Batch"):
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    try:
        results = client.fetch_weather_batch(city_list)
        for city, weather in results.items():
            st.write(f"{city}: {weather}")
    except Exception as e:
        st.error(str(e))

st.header("Green AI Energy Report")
if st.button("Get Energy Report"):
    try:
        report = client.get_energy_report()
        st.json(report)
    except Exception as e:
        st.error(str(e))

st.header("Greeting")
name = st.text_input("Name for Greeting", "Uday")
if st.button("Get Greeting"):
    try:
        greeting = client.get_greeting(name)
        st.success(greeting)
    except Exception as e:
        st.error(str(e))
