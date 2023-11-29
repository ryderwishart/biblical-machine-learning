function createDOMElement(tag, attributes, ...children) {
  const element = document.createElement(tag);
  if (attributes) {
    for (const key in attributes) {
      if (key.startsWith('on') && typeof attributes[key] === 'function') {
        element.addEventListener(key.substring(2).toLowerCase(), attributes[key]);
      } else {
        element.setAttribute(key, attributes[key]);
      }
    }
  }
  children.forEach(child => {
    if (typeof child === 'string') {
      element.appendChild(document.createTextNode(child));
    } else if (child instanceof HTMLElement) {
      element.appendChild(child);
    }
  });
  return element;
}

function parseCsvData(csvData) {
  // Parse the CSV data using papaparse
  const parsedData = Papa.parse(csvData, { header: true });

  // Extract specific data from the parsed CSV data
  const extractedData = parsedData.data.map(row => ({
    VrefID: row['vref'],
    ProtasisString: row['Protasis'],
    ProtasisPrediction: row['Protasis Gloss'],
    ApodosisString: row['Apodosis'],
    ApodosisPrediction: row['Apodosis Gloss'],
    PossibleTokens: row['Possible_Tokens']
  }));

  return extractedData;
}

// TODO: change exportTableToCSV to export generated data
function exportTableToCSV(filename) {
  const table = document.querySelector('.container table'); // Select the newly created table within the container
  const rows = table.querySelectorAll('tr'); // Select all the rows in the table
  let csvContent = '';

  rows.forEach(row => {
    const rowData = [];
    row.querySelectorAll('td').forEach(cell => { // Only select td elements for data cells
      rowData.push(cell.textContent);
    });
    csvContent += rowData.join(',') + '\n';
  });

  // Create a blob with the CSV content
  const blob = new Blob([csvContent], { type: 'text/csv' });

  // Create a link element to trigger the download
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
}

function App() {
  const container = createDOMElement('div', { class: 'container mx-auto p-6' });
  const heading = createDOMElement('h1', { class: 'text-xl font-bold mb-4' }, 'Protasis (p) and Apodosis (q) data table, with predicted matches');
  const button = createDOMElement('button', { id: 'export-csv', class: 'bg-gray-200 hover:bg-gray-300 text-black font-semibold py-2 px-4 rounded' }, 'Export to CSV');
  const buttonContainer = createDOMElement('div', { class: 'mb-4' }, button);
  const table = createDOMElement('table', { class: 'min-w-full bg-white' });
  const thead = createDOMElement('thead');
  const tbody = createDOMElement('tbody');
  const csvFilePath = 'Condensed Protasis Apodosis Analysis.csv';

  fetch(csvFilePath)
  .then(response => response.text())
  .then(csvData => {
    // Process the csvData as needed, for example, by parsing it into an array of objects
    const parsedData = parseCsvData(csvData); // Assuming you have a function to parse the CSV data
    // Populate the csvData array with the parsed data
    csvData = parsedData;
    // Populate the table with the extracted data, including the conversion of Possible_Tokens into buttons
    // Populate the table with the extracted data, including the conversion of Possible_Tokens into buttons
// Assuming csvData is an array of objects
// Assuming csvData is an array of objects
csvData.forEach(rowData => {
  const row = createDOMElement('tr');
  Object.entries(rowData).forEach(([key, value]) => {
    if ((key === 'Possible_Tokens' || key === 'ProtasisPrediction' || key === 'ApodosisPrediction') && typeof value === 'string' && value.startsWith("[('") && value.endsWith("')]")) {
      const listString = value.slice(3, -2); // Remove the brackets and quotes
      const list = listString.split("), (").map(item => item.replace(/'/g, '').split(', '));
      const buttonContainer = createDOMElement('div', { class: 'button-container' });
      list.forEach(tuple => {
        const button = createDOMElement('button', { 
          class: 'token-button', 
          style: 'background-color: #4CAF50; /* Green */ border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 5px;' 
        }, tuple.join(', '));
        button.addEventListener('click', () => {
          if (button.classList.contains('active')) {
            button.classList.remove('active');
            // Handle toggling off functionality
          } else {
            button.classList.add('active');
            // Handle toggling on functionality
          }
        });
        buttonContainer.appendChild(button);
      });
      row.appendChild(buttonContainer);
    } else {
      // If the value is not in the specified format, create a regular table cell
      const td = createDOMElement('td', { class: 'border px-4 py-2' }, value);
      row.appendChild(td);
    }
  });
  tbody.appendChild(row);
});
  })
  .catch(error => {
    console.error('Error fetching the .csv file:', error);
  });

  // Create table headers
  const headers = ['Vref ID', 'Original Protasis String', 'Predicted Protasis Tokens', 'Original Apodosis String', 'Predicted Apodosis Tokens', 'Possible Tokens'];
  const headerRow = createDOMElement('tr');
  headers.forEach(header => {
    const th = createDOMElement('th', { class: 'border px-4 py-2' }, header);
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  // Assuming csvData is an array of objects
  const csvData = []; // Replace with your actual data
  csvData.forEach(rowData => {
    const row = createDOMElement('tr');
    Object.values(rowData).forEach(value => {
      const td = createDOMElement('td', { class: 'border px-4 py-2' }, value);
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(heading);
  container.appendChild(buttonContainer);
  container.appendChild(table);

  button.addEventListener('click', () => {
    exportTableToCSV('table.csv');
  });

  return container;
}

// Assuming there is a root element in your HTML where you want to render the App component
const rootElement = document.getElementById('root');
rootElement.appendChild(App());