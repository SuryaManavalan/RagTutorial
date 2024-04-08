require('dotenv').config();
let { Pinecone } = require('@pinecone-database/pinecone');
let OpenAI = require("openai");
let { OpenAIEmbeddings } = require("@langchain/openai");
let { loadQAStuffChain } = require("langchain/chains");
let { Document } = require("langchain/document");
const LangchainOpenAI = require("@langchain/openai").OpenAI;


async function main() {
    const schedule = ["3/3/2024 6:00 AM - Wake Up", "3/3/2024 7:00 AM - Eat Breakfast", "3/3/2024 8:00 AM - Surrender to Canada", "3/3/2024 12:00 PM - Eat Lunch", "3/3/2024 5:00 PM - Go Home", "3/3/2024 6:00 PM - Eat Dinner", "3/3/2024 7:00 PM - Go to Sleep", "3/3/2024 8:00 PM - Dream"];

    const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        dangerouslyAllowBrowser: true
    });

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });

    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        batchSize: 100,
        model: 'text-embedding-3-small',
    });

    const indexName = 'opennote';

    const index = pc.index(indexName);

    const scheduleEmbeddings = await embeddings.embedDocuments(schedule);

    console.log("length of embeddings: " + scheduleEmbeddings.length);

    const scheduleVectors = scheduleEmbeddings.map((embedding, i) => ({
        id: schedule[i],
        values: embedding,
        metadata: {
            text: schedule[i],
        }
    }));

    await index.upsert(scheduleVectors);

    const query = "when do I plan to prostrate myself for our northern neighbor?";
    
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(query);
    
    let queryResponse = await index.query({
        vector: queryEmbedding,
        topK: 3,
        includeMetadata: true,
    });

    const concatenatedText = queryResponse.matches
        .map((match) => match.metadata.text)
        .join(" ");

    console.log(`Concatenated Text: ${concatenatedText}`);

    const llm = new LangchainOpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const chain = loadQAStuffChain(llm);

    const result = await chain.call({
        input_documents: [new Document({ pageContent: concatenatedText })],
        question: query,
    });

    console.log(`Answer: ${result.text}`);
}

main().catch(console.error);