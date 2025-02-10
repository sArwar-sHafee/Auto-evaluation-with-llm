import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
app = FastAPI()


@app.get('/', response_class=HTMLResponse)
async def view_csv():
    # Check the current working directory

    
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv('../script/logs/results.csv')
        
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("The CSV file is empty.")
        
        # Convert the DataFrame to HTML
        html_table = df.to_html(classes='table table-striped')
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the CSV file.")
    
    # Define a simple HTML template
    html_template = f'''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <title>CSV Viewer</title>
      </head>
      <body>
        <div class="container">
          <h1 class="mt-5">CSV File Content</h1>
          {html_table}
        </div>
      </body>
    </html>
    '''
    
    # Return the HTML template with the table
    return HTMLResponse(content=html_template)

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)
