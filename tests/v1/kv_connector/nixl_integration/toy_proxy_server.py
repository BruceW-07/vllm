# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_timestamp_ms():
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def format_duration_ms(start_time_ms, end_time_ms):
    """Format duration in milliseconds"""
    return end_time_ms - start_time_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        app.state.prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        app.state.decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8100])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[8200])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        
        # Record proxy start time
        proxy_start_ms = get_timestamp_ms()
        logger.info(f"[PROXY_START] request_id={request_id} "
                   f"proxy_start_ms={proxy_start_ms}")

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')

        # Send request to prefill service
        prefill_start_ms = get_timestamp_ms()
        
        response = await send_request_to_service(prefill_client_info, api,
                                                 req_data, request_id)
        
        prefill_end_ms = get_timestamp_ms()
        prefill_duration_ms = format_duration_ms(prefill_start_ms, prefill_end_ms)
        logger.info(f"[PREFILL_REQUEST_END] request_id={request_id} "
                   f"prefill_end_ms={prefill_end_ms} "
                   f"prefill_duration_ms={prefill_duration_ms}")

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        prefill_timing = {}

        # Debug: Log the response to see what's available
        if 'vllm_timing' in response_json:
            timing = response_json['vllm_timing']
            prefill_timing['prefill_queued_time'] = timing.get('queued_time')
            prefill_timing['prefill_execute_time'] = timing.get('execute_time')
            logger.info(f"[PREFILL_TIMING] request_id={request_id} "
                       f"prefill_queued_ms={timing.get('queued_time', 0) * 1000:.1f} "
                       f"prefill_execute_ms={timing.get('execute_time', 0) * 1000:.1f}")

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        # Calculate gap between prefill and decode
        decode_start_ms = get_timestamp_ms()
        prefill_to_decode_gap_ms = format_duration_ms(prefill_end_ms, decode_start_ms)
        logger.info(f"[DECODE_REQUEST_START] request_id={request_id} "
                   f"decode_start_ms={decode_start_ms} "
                   f"prefill_to_decode_gap_ms={prefill_to_decode_gap_ms}")

        # Stream response from decode service
        async def generate_stream():
            first_token_ms = None
            
            async for chunk in stream_service_response(decode_client_info,
                                                       api,
                                                       req_data,
                                                       request_id=request_id):
                current_time_ms = get_timestamp_ms()
                
                # Record first token time
                if first_token_ms is None:
                    first_token_ms = current_time_ms
                    decode_queue_time_ms = format_duration_ms(decode_start_ms, first_token_ms)
                    logger.info(f"[FIRST_TOKEN] request_id={request_id} "
                               f"first_token_ms={first_token_ms} "
                               f"decode_queue_time_ms={decode_queue_time_ms}")
                
                # Inject prefill timing into the first chunk
                if first_token_ms == current_time_ms and prefill_timing:
                    try:
                        import json

                        # Parse the chunk to inject timing info
                        chunk_str = chunk.decode('utf-8')
                        if chunk_str.startswith('data: '):
                            data_str = chunk_str[6:].strip()
                            if data_str and data_str != '[DONE]':
                                data = json.loads(data_str)
                                assert ('vllm_timing' in data and
                                        'queued_time' in data['vllm_timing']
                                        and 'execute_time'
                                        in data['vllm_timing'])
                                data['vllm_timing'] = {
                                    'prefill_queued_time': \
                                        prefill_timing['prefill_queued_time'],
                                    'prefill_execute_time': \
                                        prefill_timing['prefill_execute_time'],
                                    'decode_queued_time': \
                                        data['vllm_timing']['queued_time'],
                                    'decode_execute_time': \
                                        data['vllm_timing']['execute_time'],
                                }
                                # Reconstruct chunk
                                chunk = f"data: {json.dumps(data)}\n\n".encode(
                                )
                    except Exception as e:
                        logger.warning("Failed to inject timing info: %s", e)

                yield chunk
            
            # Log completion
            completion_time_ms = get_timestamp_ms()
            total_duration_ms = format_duration_ms(proxy_start_ms, completion_time_ms)
            
            logger.info(f"[REQUEST_COMPLETE] request_id={request_id} "
                       f"completion_time_ms={completion_time_ms} "
                       f"total_duration_ms={total_duration_ms}")

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        logger.error(f"Error occurred in disagg prefill proxy server - {api} endpoint: {e}")
        logger.error("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()
    
    # Setup logging with timestamp
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info(f"[PROXY_INIT] Starting proxy server on {global_args.host}:{global_args.port}")
    logger.info(f"[PROXY_CONFIG] Prefiller instances: {global_args.prefiller_instances}")
    logger.info(f"[PROXY_CONFIG] Decoder instances: {global_args.decoder_instances}")

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
