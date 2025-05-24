import { Document } from "@langchain/core/documents";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import axios from "axios";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { performance } from "perf_hooks";
import { RetrievalQAChain } from "langchain/chains";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import moment from "moment-timezone";

// .env 파일 로드 
dotenv.config();

// 주요 설정값 (크롤링 도메인, 프롬프트)
const config = {
  domain: "https://news.naver.com/section/100",
  query:
    "다음 Html 기사 목록을 바탕으로 중립적으로 핵심 내용을 요약해줘. 개인 의견이나 해석은 제외하고 원문 링크만 유지해줘.",
};

// 진행 단계 표시용 변수 및 시작 시간 저장
let currentStep = 1;
const startTime = performance.now();

// 각 단계별로 경과 시간 및 메시지 출력
function logTimeWriteOutStep(message: string): void {
  const elapsedTime = `[${((performance.now() - startTime) / 1000).toFixed(2)}s]`;
  const logMessage = `${elapsedTime} Step ${currentStep}: ${message}`;
  console.log(logMessage);
  currentStep += 1;
}

// 시간 포맷팅 함수
const getCurrentTime = (): string => {
  const m = moment().tz("Asia/Seoul");
  return m.format("YYYY-MM-DD HH:mm:ss");
};

async function main(): Promise<void> {
  // 1. 메인 함수 시작
  logTimeWriteOutStep("main 함수 시작");

  // 2. Open AI 임베딩 모델 선언 및 초기화
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY || "",
    batchSize: 512,
  });
  logTimeWriteOutStep("OpenAI Embeddings 선언 및 할당 완료");

  // 3. 지정 도메인에서 Html 데이터 크롤링
  const response = await axios.get(config.domain);
  logTimeWriteOutStep("html 파일 가져오기 완료");

  // 4. 경로 및 디렉토리 세팅 (현재 파일 기준 project base 경로 추출)
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.resolve(__filename);
  const projectBaseDir = path.dirname(__dirname);

  // 5. cache, record 디렉토리 생성 
  const cacheDir = path.join(projectBaseDir, "cache");
  const recordDir = path.join(projectBaseDir, "record");

  if (!fs.existsSync(cacheDir)) fs.mkdirSync(cacheDir);
  if (!fs.existsSync(recordDir)) fs.mkdirSync(recordDir);
  logTimeWriteOutStep("파일 생성 완료");

  // 6. 크롤링한 Html 원본을 cache 디렉토리에 저장
  const filePath = path.join(cacheDir, `${getCurrentTime()}-response.html`);
  fs.writeFileSync(filePath, response.data);
  logTimeWriteOutStep("response.html 파일로 저장 완료");

  // 7. Cheerio를 이용해 Html에서 li 태그 기준으로 기사 목록 데이터 추출
  const loader = new CheerioWebBaseLoader(config.domain, { selector: "li" });
  const loadedData = await loader.load();

  // 8. 기사 데이터를 청크 단위로 분할
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 100,
  });

  const splitDocs = await splitter.splitDocuments(loadedData);
  logTimeWriteOutStep("html 파일 로드 완료");

  // 9. splitDocs를 LangChain Document 객체로 변환
  const docs = splitDocs.map(
    (item) => new Document({ pageContent: item.pageContent, metadata: item.metadata })
  );
  logTimeWriteOutStep("Document 객체로 변환 완료");

  // 10. Vector Store(HNSWLib)로 변환하여 임베딩 적용
  const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
  logTimeWriteOutStep("Vector store 생성 완료");

  // 11. LLM 및 QA 체인 준비
  const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    returnSourceDocuments: true,
  });
  logTimeWriteOutStep("QA chain 준비 완료");

  // 12. 체인에 질의 프롬프트 전송 → 요약 생성
  const resp = await chain.call({ query: config.query });
  logTimeWriteOutStep(`QA 응답:\n ${resp.text}`);

  // 13. 결과 텍스트를 record 디렉토리에 저장
  const filePath2 = path.join(recordDir, `${getCurrentTime()}-record.txt`);
  fs.writeFileSync(filePath2, resp.text);
  logTimeWriteOutStep("요약 결과 저장 완료");

  // 14. 전체 프로세스 종료 
  logTimeWriteOutStep("최종 실행 완료");
}

// 메인 함수 실행
main();
