const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Is duality law in there with the resource available with you?',
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) {
    buffer += decoder.decode();

    const lines = buffer.split(/\r?\n/);

    for (const line of lines) {
      if (line.length) 
        console.log(line);
    }

    break;
  }

  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split(/\r?\n/);
  buffer = parts.pop() ?? "";

  for (const line of parts) {
    if (line.length) 
      console.log(line);
  }
}


