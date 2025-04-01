![image](https://github.com/user-attachments/assets/427b9e86-54f0-471c-ae9f-0e2e6310fbb7)

School of Software Engineering: Intelligent Systems

## Big Data Project: Transforming Scientific Articles into Speech using Apache Spark and Kafka

Submitters: 	Michael Berger	Shany Herskovits   Ofir Nahshon   Barack Samuni

#### **Abstract:**  
This project presents an end-to-end system designed to transform written scientific articles into audible speech, leveraging the capabilities of Apache Spark and Kafka. The goal is to establish a framework for an online service that converts text documents into streaming-ready audio files, enhancing accessibility for individuals unable to read from screens. By employing Apache Spark, the system efficiently handles the extensive computational demands of natural text-to-speech (TTS) models and processes large text batches. Apache Kafka supports simultaneous request handling, enabling seamless data ingestion from various textual sources. The workflow includes a comprehensive text processing pipeline involving parsing, filtering, and cleaning, followed by neural TTS model integration for converting processed text into speech. The system employs parallel processing to ensure swift responses, with the final output being aggregated, restored to its original order, and delivered via Kafka for real-time or on-demand consumption.

#### **Background:**  
Text-to-speech, also known as TTS, is a technology that converts written words into audible speech. An AI voice generator communicates with users when reading a screen is impossible or inconvenient. Text-to-speech technology opens up applications and information to be used in new ways, improving accessibility for individuals who cannot read text on a screen.

The project is an end-to-end system that converts written articles into audio files of spoken text. The system goal is to be the framework of an online service that receives text documents and outputs an audio file for streaming.  
We use Apache Spark and Kafka in this system due to:

* Spark ability to handle the significant computing power required to run natural TTS models by simultaneous multi workers (serveres) processing.  
* Spark ability to process large batches of text and convert them into speech data.  
* Kafka ability  to serve multiple requests simultaneously.

#### **Workflow overview:**

* **Data Ingestion:** Kafka streams textual data from various sources, such as user input, scientific articles database or plain text.  
* **Preprocessing and Analytics:** Spark processes the streamed data for parsing, filtering and cleaning before passing it to the TTS system.  
  ##### Text Processing Pipeline:  
  * The text goes through a LateX Parser  
  * The content is divided and indexed into manageable chunks.  
  * These chunks are converted to syllable sequences  
  * Chunks are sorted by syllable length  
  * The text is partitioned into batches based on total volume of text. For better performance, shorter sentences are processed with other shorter sentences, but in larger batches. The more uniform the length of sentences in a batch are, the less processing is wasted.

* **TTS Integration:** Neural TTS models convert the processed text into speech in a highly parallel manner, allowing for relatively short response times for long texts.   
  ##### Parallel Processing:  
  * The workflow splits into parallel processing streams  
  * Each stream is processed through a TTS model that converts text to waveforms and durations and trims the waveforms.  
      
* **Content delivery:** Through Kafka streaming the generated speech can be delivered as audio responses in real-time, sent to further processing, or stored in a content delivery system for other users to consume on demand.  
  ##### Final Output Stage:  
  * Results are aggregated into a DataFrame  
  * Original order is restored  
  * Waveform stitching of the multiple waveform results  
  * File saving and publishing

    

#### **Instaraction:**

 

