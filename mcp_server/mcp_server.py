from fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather() -> str:
    """Get current weather information"""
    return "It is sunny"

@mcp.tool()
def get_location() -> str:
    """Get current location information"""
    return "Beijing"

if __name__ == "__main__":
    mcp.run()
