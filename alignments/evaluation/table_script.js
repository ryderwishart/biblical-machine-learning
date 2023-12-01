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
    PossibleTokens: row['PossibleTokens']
  }));

  return extractedData;
}

function exportTableTextToCSV(filename, tableText) {
  const csvContent = tableText.map(row => row.map(cell => {
    if (Array.isArray(cell)) {
      return `"(${cell.join(', ')})"`;
    } else {
      return `"${cell}"`;
    }
  }).join(',')).join('\n');
  const csvContentWithHeaders = `vref,ProtasisText,ProtasisPrediction,ApodosisText,ApodosisPrediction,PossibleTokens${csvContent}`;
  const blob = new Blob([csvContentWithHeaders], { type: 'text/csv' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
}

function App() {
  const container = createDOMElement('div', { class: 'container mx-auto p-6' });
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
const columnOrder = ['VrefID', 'ProtasisString', 'ProtasisPrediction', 'ApodosisString', 'ApodosisPrediction', 'PossibleTokens'];

let selectedButton = null;
let buttons = [];

csvData.forEach((rowData, rowIndex) => {
  buttons[rowIndex] = [];
  const row = createDOMElement('tr');
  columnOrder.forEach((key, columnIndex) => {
    const value = rowData[key];
    if ((key === 'PossibleTokens' || key === 'ProtasisPrediction' || key === 'ApodosisPrediction') && typeof value === 'string' && value.startsWith("[('") && value.endsWith("')]")) {
      const listString = value.slice(3, -2); // Remove the brackets and quotes
      const list = listString.split("), (").map(item => item.replace(/'/g, '').split(', '));
      const buttonContainer = createDOMElement('div', { class: 'button-container' });
      list.forEach(tuple => {
        const button = createDOMElement('button', { 
          class: 'token-button', 
          style: 'background-color: green; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 5px;' 
        }, tuple.join(', '));
        button.addEventListener('click', () => {
          if (selectedButton) {
            selectedButton.style.backgroundColor = 'green'; // Deselect the currently selected button
          }
          button.style.backgroundColor = 'blue'; // Select the new button
          selectedButton = button; // Update the currently selected button
        });
        buttonContainer.appendChild(button);
        buttons[rowIndex][columnIndex] = button; // Add the button to the buttons array
      });
      const td = createDOMElement('td', { class: 'border px-4 py-2' });
      td.appendChild(buttonContainer);
      row.appendChild(td);
    } else {
      // If the value is not in the specified format, create a regular table cell
      const td = createDOMElement('td', { class: 'border px-4 py-2' }, value);
      row.appendChild(td);
    }
  });
  tbody.appendChild(row);
});
// Assuming the buttons array is properly populated before this event listener
const tableText = [];
const rows = table.getElementsByTagName('tr');
for (let i = 0; i < rows.length; i++) {
  const rowData = [];
  const cells = rows[i].getElementsByTagName('td');
  for (let j = 0; j < cells.length; j++) {
    const buttonContainer = cells[j].querySelector('.button-container');
    if (buttonContainer) {
      const buttonValues = Array.from(buttonContainer.children).map(button => {
        return `(${button.textContent})`;
      });
      rowData.push(buttonValues.join(','));
    } else {
      rowData.push(cells[j].innerText);
    }
  }
  tableText.push(rowData);
}
// Log the extracted table text to the console
console.log(tableText.join('\n'));

button.addEventListener('click', () => {
  exportTableTextToCSV('table.csv', tableText);
});

window.addEventListener('keydown', (event) => {
  if (selectedButton && (['z', 'x', 'c'].includes(event.key))) {
    const rowIndex = Array.from(selectedButton.closest('tr').parentElement.children).indexOf(selectedButton.closest('tr'));
    if (rowIndex !== -1) {
      const columnIndex = Array.from(selectedButton.closest('tr').children).indexOf(selectedButton.closest('td'));
      let targetColumnIndex;
      switch (event.key) {
        case 'z':
          targetColumnIndex = 2; // ProtasisPrediction column
          break;
        case 'x':
          targetColumnIndex = 4; // ApodosisPrediction column
          break;
        case 'c':
          targetColumnIndex = 5; // PossibleTokens column
          break;
        default:
          return; // Exit the function if the key is not 'z', 'x', or 'c'
      }
      const targetCell = document.querySelector(`.container table tbody tr:nth-child(${rowIndex + 1}) td:nth-child(${targetColumnIndex + 1})`);
      targetCell.appendChild(selectedButton);
    }
  }
});

// Assuming the buttons array is properly populated before this event listener
window.addEventListener('keydown', (event) => {
  if (selectedButton && ['w', 'a', 's', 'd'].includes(event.key)) {
    event.preventDefault(); // Prevent the default behavior (scrolling)
    const rowIndex = buttons.findIndex(row => row.includes(selectedButton));
    if (rowIndex !== -1) { // Check if selectedButton was found
      const columnIndex = buttons[rowIndex].indexOf(selectedButton);
      switch (event.key) {
        case 'w':
          if (rowIndex > 0) {
            buttons[rowIndex - 1][columnIndex].onclick();
          }
          break;
        case 's':
          if (rowIndex < buttons.length - 1) {
            buttons[rowIndex + 1][columnIndex].onclick();
          }
          break;
        case 'a':
          if (columnIndex > 0) {
            buttons[rowIndex][columnIndex - 1].onclick();
          }
          break;
        case 'd':
          if (columnIndex < buttons[rowIndex].length - 1) {
            buttons[rowIndex][columnIndex + 1].onclick();
          }
          break;
      }
    }
  }
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
  container.appendChild(buttonContainer);
  container.appendChild(table);

// Assuming the table variable holds the reference to your table element
const tableText = [];
const rows = table.getElementsByTagName('tr');
for (let i = 0; i < rows.length; i++) {
  const rowData = [];
  const cells = rows[i].getElementsByTagName('td');
  for (let j = 0; j < cells.length; j++) {
    rowData.push(cells[j].innerText);
  }
  tableText.push(rowData.join(', '));
}



  return container;
}

// Assuming there is a root element in your HTML where you want to render the App component
const rootElement = document.getElementById('root');
rootElement.appendChild(App());

