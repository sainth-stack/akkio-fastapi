/** API origin (no trailing slash). Uses VITE_BASE_URL unless the UI is opened by IP/hostname while env still points at localhost. */
export function resolveApiOrigin() {
  const envUrl = (import.meta.env.VITE_BASE_URL || '').replace(/\/$/, '');
  if (typeof window === 'undefined') {
    return envUrl || 'http://localhost:8000';
  }
  const { protocol, hostname } = window.location;
  const local = new Set(['localhost', '127.0.0.1', '[::1]']);
  let envHost = '';
  if (envUrl) {
    try {
      envHost = new URL(envUrl).hostname;
    } catch {
      envHost = '';
    }
  }
  if (!envUrl || (local.has(envHost) && !local.has(hostname))) {
    return `${protocol}//${hostname}:8000`;
  }
  return envUrl;
}

export const akkiourl = `${resolveApiOrigin()}/api`;
export function getFinalData(uniqueArr, isDate, length) {
  // Add early return if input array is empty or undefined
  if (!uniqueArr || uniqueArr.length === 0) {
    return [];
  }

  // Convert values to valid Date objects or numbers
  const parsedArr = uniqueArr?.map(val => {
    if (isDate) {
      const date = new Date(val);
      return isNaN(date.getTime()) ? null : date; // Return null for invalid dates
    }
    return !isNaN(val) ? Number(val) : null; // Convert valid numbers
  }).filter(val => val !== null); // Filter out invalid values

  // Sort the array and add safety check
  const sortedArr = parsedArr?.sort((a, b) => a - b);
  if (!sortedArr || sortedArr.length === 0) {
    return [];
  }

  const finalData = [];

  // Helper function to format dates
  const formatDate = date => {
    if (isDate) {
      const d = new Date(date);
      return d.toLocaleDateString('en-US'); // Use 'en-US' or any preferred locale
    }
    return date; // For numbers, return as is
  };

  // Calculate the size of each segment and ensure it's at least 1
  const segmentSize = Math.max(1, Math.ceil(sortedArr.length / length));

  // Adjust length if necessary to prevent empty segments
  const actualLength = Math.min(length, sortedArr.length);

  // Update the loop to use actualLength
  for (let i = 0; i < actualLength; i++) {
    const segmentStart = i * segmentSize;
    const segmentEnd = Math.min((i + 1) * segmentSize, sortedArr.length);

    const segmentValues = sortedArr.slice(segmentStart, segmentEnd);
    if (segmentValues.length === 0) continue;  // Skip empty segments

    const middleIndex = Math.floor(segmentValues.length / 2);
    const minValue = segmentValues[middleIndex];

    // Count occurrences in this segment range
    const countInRange = sortedArr.filter(
      val => val >= segmentValues[0] && val <= segmentValues[segmentValues.length - 1]
    ).length;

    finalData.push({
      value: isDate ? `${formatDate(minValue)}` : `${minValue}`,
      count: countInRange
    });
  }

  return finalData;
}


export const transformData = (data) => {
  const transformedData = [];

  // Get the keys (categories)
  const keys = Object.keys(data);

  // Assuming all categories have the same number of items
  for (let i = 0; i < Object.values(data[keys[0]]).length; i++) {
    const item = {};

    // Iterate through each category
    keys.forEach((key) => {
      // Get the value for the current index in each category
      const value = data[key][i];

      // Add the key-value pair to the item object
      item[key] = value;
    });

    // Push the item object to the transformed data array
    transformedData.push(item);
  }

  return transformedData;
};

export const transformData2 = (arrayData) => {
  // Initialize an object to store the transformed data
  const result = {};

  // Get all unique keys from the first object
  const keys = Object.keys(arrayData[0] || {});

  // Initialize empty objects for each key
  keys.forEach(key => {
    result[key] = {};
  });

  // Populate the data
  arrayData.forEach((item, index) => {
    keys.forEach(key => {
      result[key][index.toString()] = item[key];
    });
  });

  return result;
};

export const arrayToCSV = (data) => {
  const headers = Object.keys(data);
  const rowCount = Object.keys(data[headers[0]]).length;
  const csvRows = [headers.join(",")];
  for (let i = 0; i < rowCount; i++) {
    const row = headers.map((header) => {
      const value = data[header][i.toString()];
      if (typeof value === "string" && value.includes(",")) {
        return `"${value}"`;
      }
      return value;
    });
    csvRows.push(row.join(","));
  }

  return csvRows.join("\n");
};