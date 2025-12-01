def calculate_metrics(data):
    # Replace emojis with ASCII
    data = data.replace('😊', ':)')
    data = data.replace('😢', ':(')
    # Additional emoji replacements can be added here
    
    # Handle encoding issues for Windows
    try:
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(data)
    except IOError:
        with open('output.txt', 'w', encoding='ascii', errors='ignore') as f:
            f.write(data)
    
    return data

# Example usage:
if __name__ == '__main__':
    # Assuming 'data' is a string containing the metrics
    data = "Metrics data with emojis 😊 and more text"
    print(calculate_metrics(data))